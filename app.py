import math
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


# ---------------------------
# Synthetic data generation
# ---------------------------

def generate_chennai_customers(num_customers: int, rng: np.random.Generator) -> List[Customer]:
    """Create realistic zones across Chennai with a few dense clusters."""
    cluster_centers = [
        (13.0827, 80.2707),  # Chennai center
        (13.0058, 80.2209),  # South/central
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
    customers: List[Customer], stores: List[DarkStore], max_distance_km: float
) -> Dict[int, List[Tuple[int, float]]]:
    """Keep only customer-store pairs within the maximum service distance. We do not truncate the list here so the distance radius changes the optimization itself."""
    feasible: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(customers))}
    for i, customer in enumerate(customers):
        options = []
        for j, store in enumerate(stores):
            d = haversine_km(customer.lat, customer.lon, store.lat, store.lon)
            if d <= max_distance_km:
                options.append((j, d))
        options.sort(key=lambda item: item[1])
        feasible[i] = options
    return feasible


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
    """A CP-SAT based profit-maximization model with a max-distance service radius and normalized fixed costs."""
    model = cp_model.CpModel()

    y = {j: model.NewBoolVar(f"y_{j}") for j in range(len(stores))}
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}

    for i, options in feasible_pairs.items():
        for j, distance in options:
            x[(i, j)] = model.NewIntVar(0, customers[i].demand, f"x_{i}_{j}")

    # Demand satisfaction: customer demand can be served by a single or multiple stores, but no more than demand
    for i, customer in enumerate(customers):
        customer_vars = [x[(i, j)] for j, _ in feasible_pairs.get(i, [])]
        if customer_vars:
            model.Add(sum(customer_vars) <= customer.demand)

    # Capacity: store can service only up to capacity if opened
    for j, store in enumerate(stores):
        serving_vars = [x[(i, j)] for i in range(len(customers)) if (i, j) in x]
        if serving_vars:
            model.Add(sum(serving_vars) <= store.capacity * y[j])
            for i in range(len(customers)):
                if (i, j) in x:
                    model.Add(x[(i, j)] <= store.capacity * y[j])

    # Additional max-distance constraint: no assignment is allowed beyond the service radius
    for i, options in feasible_pairs.items():
        for j, distance in options:
            if distance > max_distance_km:
                raise ValueError("Distance constraint violated while building model.")
            model.Add(x[(i, j)] <= y[j] * customers[i].demand)

    # Maximum number of stores allowed to open
    model.Add(sum(y.values()) <= max_open_stores)
    if max_open_stores > 0:
        model.Add(sum(y.values()) >= min_open_stores)

    # Objective: maximize profit = revenue - delivery cost - fixed cost
    # Fixed costs are normalized to keep the objective balanced across thousands of demand points.
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
    solver.parameters.max_time_in_seconds = 15
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        empty_kpi = {
            "status": "Infeasible",
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
        return empty_kpi, pd.DataFrame([empty_kpi]), {"open_stores": [], "assigned_orders": []}

    # Collect solution metrics
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

    kpi_data = {
        "status": "Optimal" if status == cp_model.OPTIMAL else "Feasible",
        "profit": round(profit, 2),
        "revenue": round(total_revenue, 2),
        "delivery_cost": round(total_delivery, 2),
        "fixed_cost": round(total_fixed_cost, 2),
        "stores_opened": len(open_stores),
        "demand_served": total_demand_served,
        "service_rate": round((total_demand_served / sum(c.demand for c in customers)) * 100, 2),
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
    result_info: Dict,
    open_stores: List[int],
    assigned_orders: List[Dict],
) -> folium.Map:
    center = [CHENNAI_CENTER[0], CHENNAI_CENTER[1]]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    # Add city bounds polygon as a visual guide
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

    # Open and closed stores
    for store in stores:
        color = "#ff8a1d" if store.idx in open_stores else "#d0d0d0"
        folium.Marker(
            location=[store.lat, store.lon],
            tooltip=f"Dark Store {store.idx}",
            icon=folium.Icon(color="orange" if store.idx in open_stores else "gray", icon="store"),
        ).add_to(m)

    # Customers with assignment color by store
    assigned_by_store = {}
    for rec in assigned_orders:
        assigned_by_store.setdefault(rec["store_id"], []).append(rec)

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
        # sample assignment pins to keep map readable for large cases
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

    # Draw a few customer-to-store lines
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

st.sidebar.header("Inputs")
num_customers = st.sidebar.number_input("Number of customers", min_value=100, max_value=20000, value=10000, step=100)
num_candidates = st.sidebar.number_input("Number of candidate dark stores", min_value=10, max_value=100, value=50, step=1)
max_distance_km = st.sidebar.slider("Maximum service distance (km)", min_value=1.0, max_value=25.0, value=8.0, step=0.5)
start_button = st.sidebar.button("Run optimization")

if start_button:
    st.session_state.run_clicked = True

st.write(
    "Assumptions: Chennai city geography is approximated with a realistic latitude/longitude box, customer demand is random but clustered in major neighborhoods, and each candidate dark store has a fixed opening cost and finite daily capacity."
)

if st.session_state.run_clicked:
    rng = np.random.default_rng(num_customers + num_candidates + int(max_distance_km * 10))
    customers = generate_chennai_customers(num_customers, rng)
    stores = generate_dark_store_candidates(num_candidates, rng)
    feasible_pairs = build_feasible_pairs(customers, stores, max_distance_km)

    limit_values = [2, 5, 10, 20, 25]
    scenario_rows = []
    model_outputs = {}

    progress_bar = st.progress(0)
    for idx, limit in enumerate(limit_values):
        progress_bar.progress((idx + 1) / len(limit_values))
        kpi_data, _, result_info = solve_dark_store_problem(
            customers=customers,
            stores=stores,
            feasible_pairs=feasible_pairs,
            max_open_stores=limit,
            max_distance_km=max_distance_km,
        )
        model_outputs[limit] = result_info
        scenario_rows.append(
            {
                "Open store limit": limit,
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

    kpi_df = pd.DataFrame(scenario_rows)
    st.session_state.last_run = {
        "customers": customers,
        "stores": stores,
        "kpi_df": kpi_df,
        "model_outputs": model_outputs,
    }

    st.subheader("1) KPI summary")
    st.dataframe(kpi_df, use_container_width=True)

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

    selected_limit = int(kpi_df.loc[kpi_df["Profit"].idxmax(), "Open store limit"])
    selected_result = model_outputs[selected_limit]
    open_stores = selected_result["open_stores"]
    assigned_orders = selected_result["assigned_orders"]

    st.subheader("2) Detailed solution map")
    map_obj = draw_solution_map(
        customers=customers,
        stores=stores,
        result_info={},
        open_stores=open_stores,
        assigned_orders=assigned_orders,
    )
    st_folium(map_obj, width=1200, height=700)

    st.caption(
        "The map shows the open dark stores and a sample of customer demand assignments. The max-distance constraint is enforced by only allowing feasible customer-to-store pairs within the configured service radius."
    )
elif st.session_state.last_run is not None:
    last_run = st.session_state.last_run
    kpi_df = last_run["kpi_df"]
    model_outputs = last_run["model_outputs"]
    customers = last_run["customers"]
    stores = last_run["stores"]

    st.subheader("1) KPI summary")
    st.dataframe(kpi_df, use_container_width=True)

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

    selected_limit = int(kpi_df.loc[kpi_df["Profit"].idxmax(), "Open store limit"])
    selected_result = model_outputs[selected_limit]
    open_stores = selected_result["open_stores"]
    assigned_orders = selected_result["assigned_orders"]

    st.subheader("2) Detailed solution map")
    map_obj = draw_solution_map(
        customers=customers,
        stores=stores,
        result_info={},
        open_stores=open_stores,
        assigned_orders=assigned_orders,
    )
    st_folium(map_obj, width=1200, height=700)
else:
    st.info("Configure the inputs on the left and click Run optimization to solve the model.")
