import numpy as np
import streamlit as st
import sympy as sp
from scipy.integrate import quad
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import plotly.graph_objects as go


# ============================================================
# Streamlit setup
# ============================================================
st.set_page_config(
    page_title="Disk Method Lab",
    page_icon="🔴",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .small-note {opacity: 0.78; font-size: 0.92rem;}
        .metric-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            background: linear-gradient(180deg, rgba(250,250,250,1) 0%, rgba(244,244,244,1) 100%);
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            min-height: 90px;
        }
        .metric-label {
            font-size: 0.92rem;
            color: #5B6470;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Parsing helpers
# ============================================================
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
SAFE_FUNCS = {
    "pi": sp.pi,
    "e": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
}


def safe_parse_x(expr_str: str):
    x = sp.Symbol("x", real=True)
    local_dict = {"x": x, **SAFE_FUNCS}
    expr = parse_expr(
        expr_str,
        local_dict=local_dict,
        transformations=TRANSFORMS,
        evaluate=True,
    )
    return x, sp.simplify(expr)


def as_array(values, template):
    if np.isscalar(values):
        return np.full_like(template, float(values), dtype=float)
    return np.asarray(values, dtype=float)


# ============================================================
# Geometry helpers
# ============================================================
def cylinder_mesh_x(x0, x1, radius, axis_y=0.0, n_theta=56):
    """Closed cylinder for one disk, axis parallel to x."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    y_ring = axis_y + radius * np.cos(theta)
    z_ring = radius * np.sin(theta)

    x = np.concatenate([
        np.full(n_theta, x0),
        np.full(n_theta, x1),
        [x0, x1],
    ])
    y = np.concatenate([y_ring, y_ring, [axis_y, axis_y]])
    z = np.concatenate([z_ring, z_ring, [0.0, 0.0]])

    front_center = 2 * n_theta
    back_center = 2 * n_theta + 1

    I, J, K = [], [], []

    for k in range(n_theta):
        kn = (k + 1) % n_theta

        # side wall
        I.extend([k, k])
        J.extend([kn, n_theta + kn])
        K.extend([n_theta + kn, n_theta + k])

        # front cap
        I.append(front_center)
        J.append(kn)
        K.append(k)

        # back cap
        I.append(back_center)
        J.append(n_theta + k)
        K.append(n_theta + kn)

    return x, y, z, I, J, K


def exact_surface_mesh(xs, radii, axis_y=0.0, n_theta=80):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    X = np.tile(xs, (theta.size, 1))
    TH = np.tile(theta.reshape(-1, 1), (1, xs.size))
    RR = np.tile(radii, (theta.size, 1))
    Y = axis_y + RR * np.cos(TH)
    Z = RR * np.sin(TH)
    return X, Y, Z


def rim_polyline(x_positions, radii, axis_y=0.0, n_theta=72):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    X, Y, Z = [], [], []
    for xpos, radius in zip(x_positions, radii):
        X.extend(np.full(theta.size, xpos).tolist())
        Y.extend((axis_y + radius * np.cos(theta)).tolist())
        Z.extend((radius * np.sin(theta)).tolist())
        X.append(None)
        Y.append(None)
        Z.append(None)
    return X, Y, Z


def style_pack(style_name: str):
    if style_name == "Bold red (recommended)":
        return {
            "disk_color": "#D10000",
            "rim_color": "rgba(80,0,0,0.95)",
            "curve_color": "#111111",
            "region_fill": "rgba(230, 35, 35, 0.24)",
            "sample_color": "#00C853",
            "partition_color": "rgba(220, 40, 40, 0.48)",
            "axis2d": "#E31B1B",
            "axis3d": "#CFCFCF",
            "ghost_color": "#F8B4B4",
            "scene_bg": "#080808",
            "paper_bg": "#080808",
            "font_color": "#F3F4F6",
            "grid_color": "rgba(255,255,255,0.12)",
        }
    return {
        "disk_color": "#D9A441",
        "rim_color": "rgba(91,65,24,0.88)",
        "curve_color": "#2563EB",
        "region_fill": "rgba(74, 144, 226, 0.18)",
        "sample_color": "#D97706",
        "partition_color": "rgba(217,119,6,0.55)",
        "axis2d": "#374151",
        "axis3d": "#111827",
        "ghost_color": "#60A5FA",
        "scene_bg": "#FFFFFF",
        "paper_bg": "#FFFFFF",
        "font_color": "#111827",
        "grid_color": "rgba(0,0,0,0.10)",
    }


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Disk Method Controls")
    st.caption("This version only does the disk method for a region between y = f(x) and y = c, revolved around the horizontal line y = c.")

    f_str = st.text_input("Curve  y = f(x)", value="sqrt(x)")

    st.subheader("Interval")
    c1, c2 = st.columns(2)
    with c1:
        a = st.number_input("a", value=0.0, format="%.4f")
    with c2:
        b = st.number_input("b", value=4.0, format="%.4f")

    axis_c = st.number_input("Axis of rotation  y = c", value=0.0, format="%.4f")

    st.divider()
    st.subheader("Approximation")
    n_disks = st.slider("Number of disks, n", min_value=2, max_value=60, value=8, step=1)
    sample_rule = st.selectbox(
        "Radius sampled at",
        ["Midpoint", "Left endpoint", "Right endpoint"],
        index=0,
    )

    st.divider()
    st.subheader("3D appearance")
    style_name = st.selectbox(
        "Color/style",
        ["Bold red (recommended)", "Textbook gold"],
        index=0,
    )
    visual_gap_pct = st.slider(
        "Visual separation between disks",
        min_value=0,
        max_value=20,
        value=8,
        step=1,
        help="This only separates the disks visually. The volume calculation still uses the full Δx.",
    )
    n_theta = st.slider("Disk roundness", 32, 120, 72, 8)
    show_rims = st.selectbox("Emphasize disk edges", ["Yes", "No"], index=0) == "Yes"
    show_true_surface = st.selectbox("Show true smooth solid", ["No", "Yes"], index=0) == "Yes"
    show_partitions = st.selectbox("Show partition lines in 2D", ["Yes", "No"], index=0) == "Yes"

colors = style_pack(style_name)


# ============================================================
# Validate and evaluate function
# ============================================================
if np.isclose(a, b):
    st.error("Choose an interval with a ≠ b.")
    st.stop()

if b < a:
    a, b = b, a
    st.info("I swapped the endpoints so that a < b.")

try:
    x, f_expr = safe_parse_x(f_str)
    f_num = sp.lambdify(x, f_expr, modules=["numpy"])
except Exception as exc:
    st.error(f"I could not parse f(x). Details: {exc}")
    st.stop()

xs = np.linspace(a, b, 700)
try:
    f_vals = as_array(f_num(xs), xs)
except Exception as exc:
    st.error(f"f(x) could not be evaluated on [{a}, {b}]. Details: {exc}")
    st.stop()

if not np.all(np.isfinite(f_vals)):
    st.error("f(x) is not finite everywhere on the chosen interval. Adjust f(x), a, or b.")
    st.stop()

true_radii = np.abs(f_vals - axis_c)


# ============================================================
# Disk Riemann sum
# ============================================================
edges = np.linspace(a, b, n_disks + 1)
dx = (b - a) / n_disks
lefts = edges[:-1]
rights = edges[1:]
centers = (lefts + rights) / 2

if sample_rule == "Left endpoint":
    x_star = lefts
elif sample_rule == "Right endpoint":
    x_star = rights
else:
    x_star = centers

try:
    f_star = as_array(f_num(x_star), x_star)
except Exception as exc:
    st.error(f"The sample points could not be evaluated. Details: {exc}")
    st.stop()

if not np.all(np.isfinite(f_star)):
    st.error("At least one disk sample point gives a non-finite value.")
    st.stop()

radii = np.abs(f_star - axis_c)
disk_volumes = np.pi * radii**2 * dx
approx_volume = float(np.sum(disk_volumes))


def integrand(t):
    y_val = float(np.asarray(f_num(t)))
    return np.pi * (y_val - axis_c) ** 2


try:
    exact_volume_num, quad_err = quad(integrand, a, b, limit=300)
except Exception as exc:
    st.error(f"The volume integral failed numerically. Details: {exc}")
    st.stop()

try:
    sym_volume = sp.integrate(sp.pi * (f_expr - axis_c) ** 2, (x, a, b))
    if sym_volume.has(sp.Integral):
        sym_volume = None
    else:
        sym_volume = sp.simplify(sym_volume)
except Exception:
    sym_volume = None

abs_error = abs(approx_volume - exact_volume_num)
rel_error = abs_error / abs(exact_volume_num) if not np.isclose(exact_volume_num, 0.0) else np.nan


# ============================================================
# Header
# ============================================================
st.title("Disk Method Lab")
st.markdown(
    "Move the **number of disks** slider and watch the discrete solid converge toward the true solid of revolution. "
    "Each cylinder is one term of the disk-method Riemann sum."
)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    metric_card("Δx", f"{dx:.5g}")
with mc2:
    metric_card("Disk approximation", f"{approx_volume:.7g}")
with mc3:
    metric_card("Actual volume", f"{exact_volume_num:.7g}")
with mc4:
    metric_card("Relative error", "—" if np.isnan(rel_error) else f"{100 * rel_error:.4g}%")


# ============================================================
# Main plots
# ============================================================
left_col, right_col = st.columns([0.85, 1.35], gap="large")

with left_col:
    st.subheader("1. Region and partition")
    fig2d = go.Figure()

    fig2d.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([f_vals, np.full_like(xs, axis_c)[::-1]]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            fillcolor=colors["region_fill"],
            name="Region",
            hoverinfo="skip",
        )
    )
    fig2d.add_trace(
        go.Scatter(
            x=xs,
            y=f_vals,
            mode="lines",
            line=dict(width=3, color=colors["curve_color"]),
            name="y = f(x)",
        )
    )
    fig2d.add_trace(
        go.Scatter(
            x=[a, b],
            y=[axis_c, axis_c],
            mode="lines",
            line=dict(width=3, color=colors["axis2d"]),
            name="axis y = c",
        )
    )
    fig2d.add_trace(
        go.Scatter(
            x=x_star,
            y=f_star,
            mode="markers",
            marker=dict(size=8, color=colors["sample_color"], line=dict(width=1, color="#111111")),
            name="radius samples",
            customdata=np.column_stack([np.arange(1, n_disks + 1), radii]),
            hovertemplate="Disk %{customdata[0]:.0f}<br>x* = %{x:.5g}<br>R = %{customdata[1]:.5g}<extra></extra>",
        )
    )

    if show_partitions:
        px, py = [], []
        for edge in edges:
            try:
                y_edge = float(np.asarray(f_num(edge)))
            except Exception:
                continue
            px.extend([edge, edge, None])
            py.extend([axis_c, y_edge, None])
        fig2d.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines",
                line=dict(width=1, color=colors["partition_color"]),
                name="partitions",
                hoverinfo="skip",
            )
        )

    fig2d.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="x",
        yaxis_title="y",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig2d, use_container_width=True, config={"displaylogo": False})

    st.subheader("2. The Riemann sum")
    st.latex(r"\Delta x=\frac{b-a}{n}")
    st.latex(r"V_n=\sum_{i=1}^{n}\pi\,[f(x_i^*)-c]^2\,\Delta x")
    st.caption(f"Sampling rule: {sample_rule}. Here n = {n_disks} and Δx = {dx:.6g}.")

    if sym_volume is not None:
        st.markdown("**Exact integral:**")
        st.latex(r"V=\pi\int_a^b [f(x)-c]^2\,dx=" + sp.latex(sym_volume))
    else:
        st.markdown("**Actual volume (numeric integral):**")
        st.latex(r"V=\pi\int_a^b [f(x)-c]^2\,dx")
        st.caption(f"Numerical integration error estimate: ±{quad_err:.2g}")

with right_col:
    st.subheader(f"3. Approximation by {n_disks} disks")
    fig3d = go.Figure()

    if show_true_surface:
        Xg, Yg, Zg = exact_surface_mesh(xs, true_radii, axis_y=axis_c, n_theta=max(64, n_theta))
        fig3d.add_trace(
            go.Surface(
                x=Xg,
                y=Yg,
                z=Zg,
                surfacecolor=np.zeros_like(Xg),
                colorscale=[[0, colors["ghost_color"]], [1, colors["ghost_color"]]],
                showscale=False,
                opacity=0.18,
                hoverinfo="skip",
                name="True solid",
            )
        )

    visual_gap = (visual_gap_pct / 100.0) * dx
    disk_centers_vis = []

    for i, (xL, xR, R) in enumerate(zip(lefts, rights, radii), start=1):
        x0_vis = xL + visual_gap / 2
        x1_vis = xR - visual_gap / 2
        if x1_vis <= x0_vis:
            x0_vis, x1_vis = xL, xR

        Xv, Yv, Zv, I, J, K = cylinder_mesh_x(
            x0_vis,
            x1_vis,
            float(R),
            axis_y=axis_c,
            n_theta=n_theta,
        )

        fig3d.add_trace(
            go.Mesh3d(
                x=Xv,
                y=Yv,
                z=Zv,
                i=I,
                j=J,
                k=K,
                color=colors["disk_color"],
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.42, diffuse=0.95, specular=1.0, roughness=0.18, fresnel=0.12),
                lightposition=dict(x=120, y=80, z=130),
                hoverinfo="skip",
                showlegend=False,
                name=f"Disk {i}",
            )
        )
        disk_centers_vis.append((x0_vis + x1_vis) / 2)

    if show_rims:
        rim_x_positions = []
        rim_radii = []
        for xL, xR, R in zip(lefts, rights, radii):
            x0_vis = xL + visual_gap / 2
            x1_vis = xR - visual_gap / 2
            if x1_vis <= x0_vis:
                x0_vis, x1_vis = xL, xR
            rim_x_positions.extend([x0_vis, x1_vis])
            rim_radii.extend([float(R), float(R)])

        RX, RY, RZ = rim_polyline(rim_x_positions, rim_radii, axis_y=axis_c, n_theta=max(48, n_theta))
        fig3d.add_trace(
            go.Scatter3d(
                x=RX,
                y=RY,
                z=RZ,
                mode="lines",
                line=dict(width=4, color=colors["rim_color"]),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    axis_pad = 0.04 * (b - a)
    fig3d.add_trace(
        go.Scatter3d(
            x=[a - axis_pad, b + axis_pad],
            y=[axis_c, axis_c],
            z=[0, 0],
            mode="lines",
            line=dict(width=5, color=colors["axis3d"]),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    hover_text = [
        f"Disk {i+1}<br>Interval [{lefts[i]:.5g}, {rights[i]:.5g}]"
        f"<br>x* = {x_star[i]:.5g}"
        f"<br>R = {radii[i]:.5g}"
        f"<br>ΔV = {disk_volumes[i]:.5g}"
        for i in range(n_disks)
    ]
    fig3d.add_trace(
        go.Scatter3d(
            x=disk_centers_vis,
            y=np.full(n_disks, axis_c),
            z=np.zeros(n_disks),
            mode="markers",
            marker=dict(size=5, color="rgba(255,255,255,0.001)"),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    scene_font = dict(color=colors["font_color"])
    fig3d.update_layout(
        height=680,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        paper_bgcolor=colors["paper_bg"],
        font=scene_font,
        scene=dict(
            xaxis_title="x",
            yaxis_title="radial y",
            zaxis_title="z",
            aspectmode="data",
            bgcolor=colors["scene_bg"],
            camera=dict(eye=dict(x=1.85, y=1.10, z=0.68)),
            xaxis=dict(showbackground=False, gridcolor=colors["grid_color"], zerolinecolor=colors["grid_color"], color=colors["font_color"]),
            yaxis=dict(showbackground=False, gridcolor=colors["grid_color"], zerolinecolor=colors["grid_color"], color=colors["font_color"]),
            zaxis=dict(showbackground=False, gridcolor=colors["grid_color"], zerolinecolor=colors["grid_color"], color=colors["font_color"]),
        ),
    )
    st.plotly_chart(fig3d, use_container_width=True, config={"displaylogo": False})

    if visual_gap_pct > 0:
        st.caption(
            f"The {visual_gap_pct}% gaps are only for visibility. Each disk still represents the full mathematical thickness Δx = {dx:.6g}."
        )
    st.caption(f"Current display style: {style_name}.")


# ============================================================
# Disk-by-disk data
# ============================================================
st.divider()
show_table = st.selectbox("Show disk-by-disk table", ["No", "Yes"], index=0)
if show_table == "Yes":
    rows = []
    cumulative = 0.0
    for i in range(n_disks):
        cumulative += disk_volumes[i]
        rows.append(
            {
                "Disk": i + 1,
                "left": lefts[i],
                "right": rights[i],
                "x*": x_star[i],
                "radius R": radii[i],
                "πR²Δx": disk_volumes[i],
                "cumulative volume": cumulative,
            }
        )
    st.table(rows)

st.info(
    "If the old app showed red 'Importing a module script failed' boxes, that was likely a frontend Streamlit component issue. "
    "This version avoids several of those components and uses a more robust layout."
)

