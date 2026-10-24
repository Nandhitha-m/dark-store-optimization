import math
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ortools.sat.python import cp_model
from streamlit_folium import st_folium


# ---------------------------
# Basic geography and utilities
# ---------------------------
CHENNAI_CENTER = (13.0827, 80.2707)
CHENNAI_BOUNDS = {
    "lat_min": 12.8,
    "lat_max": 13.25,
    "lon_min": 80.0,
    "lon_max": 80.37,
}


@dataclass
class Customer:
    idx: int
    lat: float
    lon: float
    demand: int


@dataclass
class DarkStore:
    idx: int
    lat: float
    lon: float
    fixed_cost: float
    capacity: int


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def empty_kpi_row(max_distance_km: float, status: str = "Infeasible", reason: str = "") -> Dict:
    return {
        "status": status,
        "reason": reason,
        "profit": 0.0,
        "revenue": 0.0,
        "delivery_cost": 0.0,
        "fixed_cost": 0.0,
        "stores_opened": 0,
        "demand_served": 0,
        "service_rate": 0.0,
        "avg_distance_km": 0.0,
        "max_distance_km": max_distance_km,
    }


def normalize_kpi_columns(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize KPI DataFrame keys so both new and old runtime paths work."""
    if kpi_df is None:
        return pd.DataFrame()

    if not isinstance(kpi_df, pd.DataFrame):
        try:
            kpi_df = pd.DataFrame([kpi_df])
        except Exception:
            return pd.DataFrame()

    rename_map = {
        "status": "Status",
        "reason": "Reason",
        "profit": "Profit",
        "revenue": "Revenue",
        "delivery_cost": "Delivery cost",
        "fixed_cost": "Fixed cost",
        "stores_opened": "Stores opened",
        "demand_served": "Demand served",
        "service_rate": "Service rate (%)",
        "avg_distance_km": "Average distance (km)",
        "max_distance_km": "Max distance (km)",
    }

    for old_name, new_name in rename_map.items():
        if old_name in kpi_df.columns and new_name not in kpi_df.columns:
            kpi_df = kpi_df.rename(columns={old_name: new_name})

    if "Open store limit" not in kpi_df.columns and "open_store_limit" in kpi_df.columns:
        kpi_df = kpi_df.rename(columns={"open_store_limit": "Open store limit"})

    return kpi_df


def safe_numeric(series: pd.Series, default: float = 0.0) -> float:
    try:
        value = pd.to_numeric(series, errors="coerce")
        if value.empty:
            return float(default)
        return float(value.iloc[0]) if len(value) == 1 else float(value.median())
    except Exception:
        return float(default)


# ---------------------------
# Synthetic data generation
# ---------------------------

def generate_chennai_customers(num_customers: int, rng: np.random.Generator) -> List[Customer]:
    """Create realistic zones across Chennai with a few dense clusters."""
    cluster_centers = [
        (13.0827, 80.2707),
        (13.0058, 80.2209),
        (13.0625, 80.2467),
        (13.1231, 80.1984),
        (13.0412, 80.2561),
        (13.1088, 80.2871),
    ]
    customers = []
    for i in range(num_customers):
        cluster_idx = rng.integers(len(cluster_centers))
        center_lat, center_lon = cluster_centers[cluster_idx]
        lat = center_lat + rng.normal(0, 0.06)
        lon = center_lon + rng.normal(0, 0.07)
        lat = min(max(lat, CHENNAI_BOUNDS["lat_min"]), CHENNAI_BOUNDS["lat_max"])
        lon = min(max(lon, CHENNAI_BOUNDS["lon_min"]), CHENNAI_BOUNDS["lon_max"])
        demand = int(rng.integers(2, 18))
        customers.append(Customer(i, float(lat), float(lon), demand))
    return customers


def generate_dark_store_candidates(
    num_candidates: int, rng: np.random.Generator
) -> List[DarkStore]:
    """Create candidate stores concentrated around realistic Chennai neighborhoods with moderate operating costs."""
    cluster_centers = [
        (13.0827, 80.2707),
        (13.0058, 80.2209),
        (13.0625, 80.2467),
        (13.1231, 80.1984),
        (13.0412, 80.2561),
        (13.1088, 80.2871),
    ]
    stores = []
    for j in range(num_candidates):
        center_lat, center_lon = cluster_centers[j % len(cluster_centers)]
        lat = center_lat + rng.normal(0, 0.05)
        lon = center_lon + rng.normal(0, 0.06)
        lat = min(max(lat, CHENNAI_BOUNDS["lat_min"]), CHENNAI_BOUNDS["lat_max"])
        lon = min(max(lon, CHENNAI_BOUNDS["lon_min"]), CHENNAI_BOUNDS["lon_max"])
        capacity = int(rng.integers(500, 1800))
        fixed_cost = float(rng.uniform(30000, 120000))
        stores.append(DarkStore(j, lat, lon, fixed_cost, capacity))
    return stores


def build_feasible_pairs(
    customers: List[Customer],
    stores: List[DarkStore],
    max_distance_km: float,
    max_options_per_customer: int = 12,
) -> Dict[int, List[Tuple[int, float]]]:
    """Keep only customer-store pairs within distance. Cap the number of options per customer to avoid solver blow-ups."""
    feasible: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(customers))}
    limit = max(1, int(max_options_per_customer))

    for i, customer in enumerate(customers):
        options = []
        for j, store in enumerate(stores):
            d = haversine_km(customer.lat, customer.lon, store.lat, store.lon)
            if d <= max_distance_km:
                options.append((j, float(d)))
        options.sort(key=lambda item: item[1])
        if len(options) > limit:
            options = options[:limit]
        feasible[i] = options
    return feasible


def determine_solver_failure_reason(
    customers: List[Customer],
    stores: List[DarkStore],
    feasible_pairs: Dict[int, List[Tuple[int, float]]],
    max_open_stores: int,
    max_distance_km: float,
) -> str:
    if not customers or not stores:
        return "No customer or store data was generated."

    if max_open_stores <= 0:
        return "The model was asked to open zero stores, which cannot satisfy positive demand."

    total_demand = sum(c.demand for c in customers)
    if total_demand <= 0:
        return "Generated demand is zero, so there is nothing to serve."

    feasible_total = sum(len(v) for v in feasible_pairs.values())
    if feasible_total == 0:
        return (
            f"No feasible customer-to-store assignments exist within {max_distance_km:.1f} km. "
            "Try increasing the maximum service distance or lowering the candidate-store density."
        )

    if max_open_stores > len(stores):
        return f"The model attempted to open {max_open_stores} stores, but only {len(stores)} stores exist."

    return (
        "The CP-SAT solver reported an infeasible assignment. This usually happens when the service radius is too small, "
        "demand exceeds the usable capacity, or the selected store-limit configuration cannot cover the demand constraints."
    )


# ---------------------------
# Optimize
# ---------------------------

def solve_dark_store_problem(
    customers: List[Customer],
    stores: List[DarkStore],
    feasible_pairs: Dict[int, List[Tuple[int, float]]],
    max_open_stores: int,
    max_distance_km: float,
    unit_revenue: float = 120.0,
    delivery_cost_per_km: float = 1.0,
    fixed_cost_normalizer: float = 1000.0,
    min_open_stores: int = 1,
) -> Tuple[Dict, pd.DataFrame, Dict]:
    """CP-SAT optimization with robust failure reporting for infeasible scenarios."""
    if not customers or not stores:
        empty = empty_kpi_row(max_distance_km, status="No data", reason="No customer or store data was generated.")
        return empty, pd.DataFrame([empty]), {"open_stores": [], "assigned_orders": []}

    if max_open_stores < min_open_stores:
        max_open_stores = min_open_stores

    if max_open_stores > len(stores):
        max_open_stores = len(stores)

    total_demand = sum(c.demand for c in customers)
    if total_demand <= 0:
        empty = empty_kpi_row(max_distance_km, status="No demand", reason="Generated demand is zero.")
        return empty, pd.DataFrame([empty]), {"open_stores": [], "assigned_orders": []}

    if not any(feasible_pairs.values()):
        reason = determine_solver_failure_reason(customers, stores, feasible_pairs, max_open_stores, max_distance_km)
        empty = empty_kpi_row(max_distance_km, status="Infeasible", reason=reason)
        return empty, pd.DataFrame([empty]), {"open_stores": [], "assigned_orders": []}

    model = cp_model.CpModel()

    y = {j: model.NewBoolVar(f"y_{j}") for j in range(len(stores))}
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}

    for i, options in feasible_pairs.items():
        for j, _ in options:
            x[(i, j)] = model.NewIntVar(0, customers[i].demand, f"x_{i}_{j}")

    for i, customer in enumerate(customers):
        customer_vars = [x[(i, j)] for j, _ in feasible_pairs.get(i, [])]
        if customer_vars:
            model.Add(sum(customer_vars) <= customer.demand)

    for j, store in enumerate(stores):
        serving_vars = [x[(i, j)] for i in range(len(customers)) if (i, j) in x]
        if serving_vars:
            model.Add(sum(serving_vars) <= store.capacity * y[j])
            for i in range(len(customers)):
                if (i, j) in x:
                    model.Add(x[(i, j)] <= store.capacity * y[j])

    for i, options in feasible_pairs.items():
        for j, distance in options:
            if distance > max_distance_km:
                raise ValueError("Distance constraint violated while building model.")
            model.Add(x[(i, j)] <= y[j] * customers[i].demand)

    model.Add(sum(y.values()) <= max_open_stores)
    if max_open_stores > 0:
        model.Add(sum(y.values()) >= min_open_stores)

    revenue_terms = []
    delivery_terms = []
    fixed_terms = []

    for (i, j), var in x.items():
        _, distance = next((pair for pair in feasible_pairs[i] if pair[0] == j), (None, None))
        if distance is None:
            continue
        revenue_terms.append(unit_revenue * var)
        delivery_terms.append((delivery_cost_per_km * distance) * var)

    for j, store in enumerate(stores):
        fixed_terms.append((store.fixed_cost / fixed_cost_normalizer) * y[j])

    objective = sum(revenue_terms) - sum(delivery_terms) - sum(fixed_terms)
    model.Maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        reason = determine_solver_failure_reason(customers, stores, feasible_pairs, max_open_stores, max_distance_km)
        empty = empty_kpi_row(max_distance_km, status="Infeasible", reason=reason)
        return empty, pd.DataFrame([empty]), {"open_stores": [], "assigned_orders": []}

    open_stores = [j for j in range(len(stores)) if solver.Value(y[j])]
    assigned_orders = []
    total_revenue = 0.0
    total_delivery = 0.0
    total_fixed_cost = 0.0
    total_demand_served = 0
    average_distance = 0.0

    for (i, j), var in x.items():
        assigned_units = solver.Value(var)
        if assigned_units <= 0:
            continue
        _, distance = next((pair for pair in feasible_pairs[i] if pair[0] == j), (None, None))
        total_revenue += assigned_units * unit_revenue
        total_delivery += assigned_units * delivery_cost_per_km * distance
        total_demand_served += assigned_units
        average_distance += distance * assigned_units
        assigned_orders.append(
            {
                "customer_id": i,
                "store_id": j,
                "assigned_units": assigned_units,
                "distance_km": round(distance, 2),
            }
        )

    for j in open_stores:
        total_fixed_cost += stores[j].fixed_cost

    if total_demand_served > 0:
        average_distance = average_distance / total_demand_served
    else:
        average_distance = 0.0

    profit = total_revenue - total_delivery - total_fixed_cost
    service_rate = (total_demand_served / total_demand) * 100.0 if total_demand > 0 else 0.0

    kpi_data = {
        "status": "Optimal" if status == cp_model.OPTIMAL else "Feasible",
        "reason": "Feasible solution found within the chosen store and distance limits.",
        "profit": round(profit, 2),
        "revenue": round(total_revenue, 2),
        "delivery_cost": round(total_delivery, 2),
        "fixed_cost": round(total_fixed_cost, 2),
        "stores_opened": len(open_stores),
        "demand_served": total_demand_served,
        "service_rate": round(service_rate, 2),
        "avg_distance_km": round(average_distance, 2),
        "max_distance_km": max_distance_km,
    }

    kpi_df = pd.DataFrame([kpi_data])
    return kpi_data, kpi_df, {"open_stores": open_stores, "assigned_orders": assigned_orders}


# ---------------------------
# Map rendering
# ---------------------------

def draw_solution_map(
    customers: List[Customer],
    stores: List[DarkStore],
    open_stores: List[int],
    assigned_orders: List[Dict],
) -> folium.Map:
    center = [CHENNAI_CENTER[0], CHENNAI_CENTER[1]]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    city_outline = [
        [CHENNAI_BOUNDS["lat_min"], CHENNAI_BOUNDS["lon_min"]],
        [CHENNAI_BOUNDS["lat_min"], CHENNAI_BOUNDS["lon_max"]],
        [CHENNAI_BOUNDS["lat_max"], CHENNAI_BOUNDS["lon_max"]],
        [CHENNAI_BOUNDS["lat_max"], CHENNAI_BOUNDS["lon_min"]],
    ]
    folium.Polygon(city_outline, color="gray", fill=False, weight=1, opacity=0.5).add_to(m)

    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for store in stores:
        folium.Marker(
            location=[store.lat, store.lon],
            tooltip=f"Dark Store {store.idx}",
            icon=folium.Icon(color="orange" if store.idx in open_stores else "gray", icon="store"),
        ).add_to(m)

    if assigned_orders:
        if len(assigned_orders) <= 5000:
            customer_points = []
            for rec in assigned_orders:
                customer = customers[rec["customer_id"]]
                store_id = rec["store_id"]
                customer_points.append((customer.lat, customer.lon, store_id))
            for lat, lon, store_id in customer_points:
                store_color = color_palette[store_id % len(color_palette)]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=2.2,
                    color=store_color,
                    fill=True,
                    fill_color=store_color,
                    fill_opacity=0.75,
                    opacity=0.7,
                    weight=0.5,
                ).add_to(m)
        else:
            sample = assigned_orders[:500]
            for rec in sample:
                customer = customers[rec["customer_id"]]
                store_id = rec["store_id"]
                color = color_palette[store_id % len(color_palette)]
                folium.CircleMarker(
                    location=[customer.lat, customer.lon],
                    radius=2.2,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                ).add_to(m)

        for rec in assigned_orders[:2000]:
            customer = customers[rec["customer_id"]]
            store = stores[rec["store_id"]]
            folium.PolyLine(
                [(customer.lat, customer.lon), (store.lat, store.lon)],
                color="#1f77b4",
                weight=0.8,
                opacity=0.35,
            ).add_to(m)

    return m


# ---------------------------
# Streamlit app
# ---------------------------

st.set_page_config(layout="wide")

st.title("Dark Store Location Optimisation")

if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False
if "last_run" not in st.session_state:
    st.session_state.last_run = None
if "last_run_key" not in st.session_state:
    st.session_state.last_run_key = None

st.sidebar.header("Inputs")
num_customers = st.sidebar.number_input(
    "Number of customers",
    min_value=100,
    max_value=20000,
    value=10000,
    step=100,
)
num_candidates = st.sidebar.number_input(
    "Number of candidate dark stores",
    min_value=10,
    max_value=60,
    value=50,
    step=1,
)
max_distance_km = st.sidebar.slider(
    "Maximum service distance (km)",
    min_value=1.0,
    max_value=35.0,
    value=8.0,
    step=0.5,
)
start_button = st.sidebar.button("Run optimization")

current_input_key = (
    int(num_customers),
    int(num_candidates),
    float(max_distance_km),
)

if start_button:
    st.session_state.run_clicked = True
    st.session_state.last_run_key = current_input_key

st.write(
    "Assumptions: Chennai city geography is approximated with a realistic latitude/longitude box, customer demand is random but clustered in major neighborhoods, and each candidate dark store has moderate operating costs and capacity."
)

try:
    if st.session_state.run_clicked:
        try:
            safe_num_customers = int(num_customers)
            safe_num_candidates = int(num_candidates)

            if safe_num_customers > 20000:
                safe_num_customers = 20000
            if safe_num_candidates > 60:
                safe_num_candidates = 60

            max_options_per_customer = 16
            if safe_num_customers >= 15000 or safe_num_candidates >= 40:
                max_options_per_customer = 10
            elif safe_num_customers >= 8000:
                max_options_per_customer = 12

            rng = np.random.default_rng(safe_num_customers + safe_num_candidates + int(max_distance_km * 10))
            customers = generate_chennai_customers(safe_num_customers, rng)
            stores = generate_dark_store_candidates(safe_num_candidates, rng)
            feasible_pairs = build_feasible_pairs(
                customers,
                stores,
                max_distance_km,
                max_options_per_customer=max_options_per_customer,
            )

            limit_values = [2, 5, 10, 20, 25, 35, 50, 60]
            scenario_rows = []
            model_outputs = {}

            progress_bar = st.progress(0)
            for idx, limit in enumerate(limit_values):
                if limit > safe_num_candidates:
                    continue
                progress_bar.progress((idx + 1) / len(limit_values))
                try:
                    kpi_data, _, result_info = solve_dark_store_problem(
                        customers=customers,
                        stores=stores,
                        feasible_pairs=feasible_pairs,
                        max_open_stores=limit,
                        max_distance_km=max_distance_km,
                    )
                except Exception as exc:
                    kpi_data = empty_kpi_row(max_distance_km, status="Error", reason=f"Unexpected solver error: {exc}")
                    result_info = {"open_stores": [], "assigned_orders": []}

                model_outputs[limit] = result_info
                scenario_rows.append(
                    {
                        "Open store limit": limit,
                        "Status": kpi_data.get("status", "Unknown"),
                        "Reason": kpi_data.get("reason", "N/A"),
                        "Profit": kpi_data.get("profit", 0.0),
                        "Revenue": kpi_data.get("revenue", 0.0),
                        "Delivery cost": kpi_data.get("delivery_cost", 0.0),
                        "Fixed cost": kpi_data.get("fixed_cost", 0.0),
                        "Stores opened": kpi_data.get("stores_opened", 0),
                        "Demand served": kpi_data.get("demand_served", 0),
                        "Service rate (%)": kpi_data.get("service_rate", 0.0),
                        "Average distance (km)": kpi_data.get("avg_distance_km", 0.0),
                    }
                )

            if not scenario_rows:
                kpi_df = normalize_kpi_columns(pd.DataFrame([empty_kpi_row(max_distance_km, status="No feasible runs", reason="No tested store limits fit the candidate count.")]))
            else:
                kpi_df = normalize_kpi_columns(pd.DataFrame(scenario_rows))

            st.session_state.last_run = {
                "customers": customers,
                "stores": stores,
                "kpi_df": kpi_df,
                "model_outputs": model_outputs,
            }
            st.session_state.last_run_key = current_input_key
            st.session_state.run_clicked = False

            st.subheader("1) KPI summary")
            st.dataframe(kpi_df, use_container_width=True)

            feasible_rows = kpi_df[kpi_df["Status"].isin(["Optimal", "Feasible"])].copy()
            if not feasible_rows.empty:
                feasible_rows["Profit"] = pd.to_numeric(feasible_rows["Profit"], errors="coerce").fillna(0.0)
                best_row = feasible_rows.loc[feasible_rows["Profit"].idxmax()]
                best_limit = int(best_row["Open store limit"])
                st.success(
                    f"Best configuration for this run: open-store limit = {best_limit}, profit = ₹{best_row['Profit']:.2f}, "
                    f"service rate = {best_row['Service rate (%)']:.2f}%"
                )
            else:
                st.warning("No feasible solution was found for any tested store limit. See the reason column above.")

            st.subheader("Profit vs. maximum open stores")
            fig = px.line(
                kpi_df,
                x="Open store limit",
                y="Profit",
                markers=True,
                title="Profit sensitivity to the maximum number of dark stores allowed to open",
                labels={"Open store limit": "Max stores allowed to open", "Profit": "Profit (₹)"},
            )
            st.plotly_chart(fig, use_container_width=True)

            if not kpi_df.empty and "Open store limit" in kpi_df.columns:
                valid_profit_rows = kpi_df[kpi_df["Status"].isin(["Optimal", "Feasible"]) & (pd.to_numeric(kpi_df["Profit"], errors="coerce") > 0)]
                if not valid_profit_rows.empty:
                    valid_profit_rows = valid_profit_rows.copy()
                    valid_profit_rows["Profit"] = pd.to_numeric(valid_profit_rows["Profit"], errors="coerce").fillna(0.0)
                    best_limit = int(valid_profit_rows.loc[valid_profit_rows["Profit"].idxmax(), "Open store limit"])
                    selected_result = model_outputs.get(best_limit, {"open_stores": [], "assigned_orders": []})
                    open_stores = selected_result.get("open_stores", [])
                    assigned_orders = selected_result.get("assigned_orders", [])

                    st.subheader("2) Detailed solution map")
                    if open_stores or assigned_orders:
                        map_obj = draw_solution_map(customers, stores, open_stores, assigned_orders)
                        st_folium(map_obj, width=1200, height=700)
                    else:
                        st.info("No assignments were generated for the best feasible solution; the model reported a valid but empty assignment set.")
                else:
                    st.subheader("2) Detailed solution map")
                    st.info("The optimization did not find a profitable feasible solution for this input combination.")
                    last_row = kpi_df.iloc[-1]
                    st.warning(f"Failure reason: {last_row.get('Reason', 'Unknown')}")
            else:
                st.subheader("2) Detailed solution map")
                st.info("No valid optimization output is available.")

        except Exception as exc:
            st.error(f"The optimization failed unexpectedly: {exc}")
            st.code(traceback.format_exc())
            st.info("Try reducing the service distance if the model becomes infeasible, or lower the customer/store count to keep the feasibility search stable.")
            st.stop()

    elif st.session_state.last_run is not None and st.session_state.last_run_key == current_input_key:
        last_run = st.session_state.last_run
        kpi_df = normalize_kpi_columns(last_run["kpi_df"])
        model_outputs = last_run["model_outputs"]
        customers = last_run["customers"]
        stores = last_run["stores"]

        st.subheader("1) KPI summary")
        st.dataframe(kpi_df, use_container_width=True)

        feasible_rows = kpi_df[kpi_df["Status"].isin(["Optimal", "Feasible"])].copy()
        if not feasible_rows.empty:
            feasible_rows["Profit"] = pd.to_numeric(feasible_rows["Profit"], errors="coerce").fillna(0.0)
            best_row = feasible_rows.loc[feasible_rows["Profit"].idxmax()]
            best_limit = int(best_row["Open store limit"])
            st.success(
                f"Best configuration for this run: open-store limit = {best_limit}, profit = ₹{best_row['Profit']:.2f}, "
                f"service rate = {best_row['Service rate (%)']:.2f}%"
            )
        else:
            st.warning("No feasible solution was found for any tested store limit. See the reason column above.")

        st.subheader("Profit vs. maximum open stores")
        fig = px.line(
            kpi_df,
            x="Open store limit",
            y="Profit",
            markers=True,
            title="Profit sensitivity to the maximum number of dark stores allowed to open",
            labels={"Open store limit": "Max stores allowed to open", "Profit": "Profit (₹)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        valid_profit_rows = kpi_df[kpi_df["Status"].isin(["Optimal", "Feasible"]) & (pd.to_numeric(kpi_df["Profit"], errors="coerce") > 0)]
        if not valid_profit_rows.empty:
            valid_profit_rows = valid_profit_rows.copy()
            valid_profit_rows["Profit"] = pd.to_numeric(valid_profit_rows["Profit"], errors="coerce").fillna(0.0)
            best_limit = int(valid_profit_rows.loc[valid_profit_rows["Profit"].idxmax(), "Open store limit"])
            selected_result = model_outputs.get(best_limit, {"open_stores": [], "assigned_orders": []})
            open_stores = selected_result.get("open_stores", [])
            assigned_orders = selected_result.get("assigned_orders", [])

            st.subheader("2) Detailed solution map")
            if open_stores or assigned_orders:
                map_obj = draw_solution_map(customers, stores, open_stores, assigned_orders)
                st_folium(map_obj, width=1200, height=700)
            else:
                st.info("No assignments were generated for the best feasible solution; the model reported a valid but empty assignment set.")
        else:
            st.subheader("2) Detailed solution map")
            st.info("The optimization did not find a profitable feasible solution for this input combination.")
            if "Reason" in kpi_df.columns and not kpi_df.empty:
                reason = kpi_df.iloc[-1].get("Reason", "Unknown")
                st.warning(f"Failure reason: {reason}")
    else:
        st.info("Configure the inputs on the left and click Run optimization to solve the model.")
except Exception as exc:
    st.error("Unexpected application error. Please check the input values and try again.")
    st.code(traceback.format_exc())
    st.stop()
