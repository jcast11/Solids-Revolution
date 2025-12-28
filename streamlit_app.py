import numpy as np
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from scipy.integrate import quad
import plotly.graph_objects as go

# -----------------------------
# Helpers
# -----------------------------
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
    "Max": sp.Max,
    "Min": sp.Min,
}

def safe_parse(expr_str: str, varname: str):
    v = sp.Symbol(varname, real=True)
    local_dict = {varname: v, **SAFE_FUNCS}
    expr = parse_expr(expr_str, local_dict=local_dict, transformations=TRANSFORMS, evaluate=True)
    return v, sp.simplify(expr)

def make_numeric(var, expr):
    return sp.lambdify(var, expr, modules=["numpy"])

def radii_from_two_curves_about_line(curve1_vals, curve2_vals, axis_c):
    r1 = np.abs(curve1_vals - axis_c)
    r2 = np.abs(curve2_vals - axis_c)
    R = np.maximum(r1, r2)
    r = np.minimum(r1, r2)
    return R, r

# ---- numeric volumes ----

def vol_washers_dx_about_yc(f_num, g_num, a, b, c):
    def integrand(x):
        fx, gx = f_num(x), g_num(x)
        R, r = radii_from_two_curves_about_line(fx, gx, c)
        return np.pi * (R**2 - r**2)
    return quad(integrand, a, b, limit=300)

def vol_washers_dy_about_xc(f_num, g_num, y1, y2, c):
    def integrand(y):
        fx, gx = f_num(y), g_num(y)
        R, r = radii_from_two_curves_about_line(fx, gx, c)
        return np.pi * (R**2 - r**2)
    return quad(integrand, y1, y2, limit=300)

def vol_shells_dx_about_xc(f_num, g_num, a, b, c):
    def integrand(x):
        radius = np.abs(x - c)
        height = np.abs(f_num(x) - g_num(x))
        return 2*np.pi * radius * height
    return quad(integrand, a, b, limit=300)

def vol_shells_dy_about_yc(f_num, g_num, y1, y2, c):
    def integrand(y):
        radius = np.abs(y - c)
        height = np.abs(f_num(y) - g_num(y))
        return 2*np.pi * radius * height
    return quad(integrand, y1, y2, limit=300)

# ---- symbolic attempts (best-effort) ----
# Note: Abs/Max/Min can defeat closed forms; numeric result is the reliable fallback.

def sym_washers_dx_about_yc(x, f, g, a, b, c):
    integrand = sp.pi * (sp.Max((f-c)**2, (g-c)**2) - sp.Min((f-c)**2, (g-c)**2))
    try:
        out = sp.integrate(integrand, (x, a, b))
        return None if out.has(sp.Integral) else sp.simplify(out)
    except Exception:
        return None

def sym_washers_dy_about_xc(y, f, g, y1, y2, c):
    integrand = sp.pi * (sp.Max((f-c)**2, (g-c)**2) - sp.Min((f-c)**2, (g-c)**2))
    try:
        out = sp.integrate(integrand, (y, y1, y2))
        return None if out.has(sp.Integral) else sp.simplify(out)
    except Exception:
        return None

def sym_shells_dx_about_xc(x, f, g, a, b, c):
    integrand = 2*sp.pi * sp.Abs(x-c) * sp.Abs(f-g)
    try:
        out = sp.integrate(integrand, (x, a, b))
        return None if out.has(sp.Integral) else sp.simplify(out)
    except Exception:
        return None

def sym_shells_dy_about_yc(y, f, g, y1, y2, c):
    integrand = 2*sp.pi * sp.Abs(y-c) * sp.Abs(f-g)
    try:
        out = sp.integrate(integrand, (y, y1, y2))
        return None if out.has(sp.Integral) else sp.simplify(out)
    except Exception:
        return None

# ---- 3D mesh helpers ----

def surface_mesh_from_radii_xparam(xs, R, axis_y, n_theta=160):
    thetas = np.linspace(0, 2*np.pi, n_theta)
    X = np.tile(xs, (n_theta, 1))
    TH = np.tile(thetas.reshape(-1, 1), (1, xs.size))
    RR = np.tile(R, (n_theta, 1))
    Y = axis_y + RR * np.cos(TH)
    Z = RR * np.sin(TH)
    return X, Y, Z

def surface_mesh_from_radii_yparam(ys, R, axis_x, n_theta=160):
    thetas = np.linspace(0, 2*np.pi, n_theta)
    Y = np.tile(ys, (n_theta, 1))
    TH = np.tile(thetas.reshape(-1, 1), (1, ys.size))
    RR = np.tile(R, (n_theta, 1))
    X = axis_x + RR * np.cos(TH)
    Z = RR * np.sin(TH)
    return X, Y, Z

def washer_slice_circles_x(xs_slices, R, r, axis_y):
    theta = np.linspace(0, 2*np.pi, 160)
    traces = []
    for xi, Ro, ri in zip(xs_slices, R, r):
        y_out = axis_y + Ro*np.cos(theta); z_out = Ro*np.sin(theta); x_out = np.full_like(theta, xi)
        y_in  = axis_y + ri*np.cos(theta); z_in  = ri*np.sin(theta); x_in  = np.full_like(theta, xi)
        traces.append(go.Scatter3d(x=x_out, y=y_out, z=z_out, mode="lines", showlegend=False))
        traces.append(go.Scatter3d(x=x_in,  y=y_in,  z=z_in,  mode="lines", showlegend=False))
    return traces

def washer_slice_circles_y(ys_slices, R, r, axis_x):
    theta = np.linspace(0, 2*np.pi, 160)
    traces = []
    for yi, Ro, ri in zip(ys_slices, R, r):
        x_out = axis_x + Ro*np.cos(theta); z_out = Ro*np.sin(theta); y_out = np.full_like(theta, yi)
        x_in  = axis_x + ri*np.cos(theta); z_in  = ri*np.sin(theta); y_in  = np.full_like(theta, yi)
        traces.append(go.Scatter3d(x=x_out, y=y_out, z=z_out, mode="lines", showlegend=False))
        traces.append(go.Scatter3d(x=x_in,  y=y_in,  z=z_in,  mode="lines", showlegend=False))
    return traces

def shell_preview_circles_about_xc(xs_slices, f_num, g_num, axis_x):
    theta = np.linspace(0, 2*np.pi, 160)
    traces = []
    for xi in xs_slices:
        radius = float(np.abs(xi-axis_x))
        midy = float((f_num(xi)+g_num(xi))/2.0)
        x = axis_x + radius*np.cos(theta)
        z = radius*np.sin(theta)
        y = np.full_like(theta, midy)
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=False))
    return traces

def shell_preview_circles_about_yc(ys_slices, f_num, g_num, axis_y):
    theta = np.linspace(0, 2*np.pi, 160)
    traces = []
    for yi in ys_slices:
        radius = float(np.abs(yi-axis_y))
        midx = float((f_num(yi)+g_num(yi))/2.0)
        x = np.full_like(theta, midx)
        z = radius*np.sin(theta)
        y = axis_y + radius*np.cos(theta)
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=False))
    return traces

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Solids of Revolution: Washers + Shells (2 curves)", layout="wide")
st.title("Solid of Revolution — Two Curves (Washers + Shell Method)")

with st.sidebar:
    st.header("Choose method")
    mode = st.selectbox(
        "Method / Axis",
        [
            "Washers: Horizontal axis y=c (dx), enter y=f(x), y=g(x)",
            "Washers: Vertical axis x=c (dy), enter x=f(y), x=g(y)",
            "Shells:  Vertical axis x=c (dx), enter y=f(x), y=g(x)",
            "Shells:  Horizontal axis y=c (dy), enter x=f(y), x=g(y)",
        ],
    )

    st.divider()
    st.subheader("Functions (2 boundaries)")
    if mode in (
        "Washers: Horizontal axis y=c (dx), enter y=f(x), y=g(x)",
        "Shells:  Vertical axis x=c (dx), enter y=f(x), y=g(x)",
    ):
        st.caption("Enter curves as **y = f(x)** and **y = g(x)**")
        f_str = st.text_input("f(x) =", value="sqrt(x)")
        g_str = st.text_input("g(x) =", value="0")
    else:
        st.caption("Enter boundaries as **x = f(y)** and **x = g(y)**")
        f_str = st.text_input("f(y) =", value="y**2")
        g_str = st.text_input("g(y) =", value="0")

    st.divider()
    st.subheader("Interval")
    col1, col2 = st.columns(2)
    if "dy" in mode:
        with col1:
            a = st.number_input("y₁", value=0.0)
        with col2:
            b = st.number_input("y₂", value=1.0)
    else:
        with col1:
            a = st.number_input("a", value=0.0)
        with col2:
            b = st.number_input("b", value=4.0)

    st.divider()
    st.subheader("Axis of rotation")
    if "y=c" in mode:
        axis_c = st.number_input("Rotate around y = c", value=0.0)
    else:
        axis_c = st.number_input("Rotate around x = c", value=0.0)

    st.divider()
    st.subheader("Plot quality")
    n_param = st.slider("Samples along interval", 150, 1000, 500, 25)
    n_theta = st.slider("Surface theta resolution", 80, 320, 180, 10)
    n_slices = st.slider("Slice preview (count)", 3, 30, 10, 1)

# Validate interval
if b == a:
    st.error("Please choose an interval with b ≠ a.")
    st.stop()
if b < a:
    a, b = b, a
    st.warning("Swapped endpoints so the interval is increasing.")

# Parse + numeric funcs
try:
    if "enter y=f(x)" in mode:
        x, f_expr = safe_parse(f_str, "x")
        _, g_expr = safe_parse(g_str, "x")
        f_num = make_numeric(x, f_expr)
        g_num = make_numeric(x, g_expr)
        var = x
    else:
        y, f_expr = safe_parse(f_str, "y")
        _, g_expr = safe_parse(g_str, "y")
        f_num = make_numeric(y, f_expr)
        g_num = make_numeric(y, g_expr)
        var = y
except Exception as e:
    st.error(f"Could not parse one of the functions. Details: {e}")
    st.stop()

# Sample + sanity check
param = np.linspace(a, b, int(n_param))
try:
    v1 = f_num(param)
    v2 = g_num(param)
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        st.error("One function produced non-finite values on the interval. Adjust the interval or functions.")
        st.stop()
except Exception as e:
    st.error(f"Functions failed to evaluate numerically. Details: {e}")
    st.stop()

# Compute volume + symbolic attempt
if mode.startswith("Washers: Horizontal"):
    R_arr, r_arr = radii_from_two_curves_about_line(v1, v2, axis_c)
    vol_num, vol_err = vol_washers_dx_about_yc(f_num, g_num, a, b, axis_c)
    vol_sym = sym_washers_dx_about_yc(var, f_expr, g_expr, a, b, axis_c)

elif mode.startswith("Washers: Vertical"):
    R_arr, r_arr = radii_from_two_curves_about_line(v1, v2, axis_c)
    vol_num, vol_err = vol_washers_dy_about_xc(f_num, g_num, a, b, axis_c)
    vol_sym = sym_washers_dy_about_xc(var, f_expr, g_expr, a, b, axis_c)

elif mode.startswith("Shells:  Vertical"):
    vol_num, vol_err = vol_shells_dx_about_xc(f_num, g_num, a, b, axis_c)
    vol_sym = sym_shells_dx_about_xc(var, f_expr, g_expr, a, b, axis_c)

else:  # Shells about horizontal axis
    vol_num, vol_err = vol_shells_dy_about_yc(f_num, g_num, a, b, axis_c)
    vol_sym = sym_shells_dy_about_yc(var, f_expr, g_expr, a, b, axis_c)

# -----------------------------
# Plots
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("2D view (region + axis)")
    fig2d = go.Figure()

    if "enter y=f(x)" in mode:
        # y=f(x), y=g(x)
        fig2d.add_trace(go.Scatter(x=param, y=v1, mode="lines", name="y = f(x)"))
        fig2d.add_trace(go.Scatter(x=param, y=v2, mode="lines", name="y = g(x)"))
        if "y=c" in mode:
            fig2d.add_trace(go.Scatter(x=[a, b], y=[axis_c, axis_c], mode="lines", name="axis y=c"))
        else:
            fig2d.add_trace(go.Scatter(x=[axis_c, axis_c], y=[min(v1.min(), v2.min()), max(v1.max(), v2.max())],
                                       mode="lines", name="axis x=c"))

        fig2d.add_trace(go.Scatter(
            x=np.concatenate([param, param[::-1]]),
            y=np.concatenate([v1, v2[::-1]]),
            fill="toself", mode="lines", name="region", opacity=0.18
        ))
        fig2d.update_layout(xaxis_title="x", yaxis_title="y", height=430,
                            margin=dict(l=10, r=10, t=10, b=10))
    else:
        # x=f(y), x=g(y)
        fig2d.add_trace(go.Scatter(x=v1, y=param, mode="lines", name="x = f(y)"))
        fig2d.add_trace(go.Scatter(x=v2, y=param, mode="lines", name="x = g(y)"))
        if "y=c" in mode:
            fig2d.add_trace(go.Scatter(x=[min(v1.min(), v2.min()), max(v1.max(), v2.max())],
                                       y=[axis_c, axis_c], mode="lines", name="axis y=c"))
        else:
            fig2d.add_trace(go.Scatter(x=[axis_c, axis_c], y=[a, b], mode="lines", name="axis x=c"))

        fig2d.add_trace(go.Scatter(
            x=np.concatenate([v1, v2[::-1]]),
            y=np.concatenate([param, param[::-1]]),
            fill="toself", mode="lines", name="region", opacity=0.18
        ))
        fig2d.update_layout(xaxis_title="x", yaxis_title="y", height=430,
                            margin=dict(l=10, r=10, t=10, b=10))

    st.plotly_chart(fig2d, use_container_width=True)

    st.subheader("Volume formula")
    if mode.startswith("Washers: Horizontal"):
        st.latex(r"V=\pi\int_a^b\left(R(x)^2-r(x)^2\right)\,dx")
    elif mode.startswith("Washers: Vertical"):
        st.latex(r"V=\pi\int_{y_1}^{y_2}\left(R(y)^2-r(y)^2\right)\,dy")
    elif mode.startswith("Shells:  Vertical"):
        st.latex(r"V=2\pi\int_a^b |x-c|\;|f(x)-g(x)|\,dx")
    else:
        st.latex(r"V=2\pi\int_{y_1}^{y_2} |y-c|\;|f(y)-g(y)|\,dy")

    st.write(f"**Numeric volume:** {vol_num:.10g}")
    st.caption(f"Estimated numeric integration error: ±{vol_err:.2g}")
    if vol_sym is not None:
        st.write("**Symbolic (exact) attempt:**")
        st.latex(sp.latex(vol_sym))
    else:
        st.caption("Symbolic integration wasn’t available here (numeric result above is reliable).")

with right:
    st.subheader("3D visualization (interactive)")
    fig3d = go.Figure()

    if mode.startswith("Washers: Horizontal"):
        R_arr, r_arr = radii_from_two_curves_about_line(v1, v2, axis_c)
        Xo, Yo, Zo = surface_mesh_from_radii_xparam(param, R_arr, axis_c, n_theta=int(n_theta))
        Xi, Yi, Zi = surface_mesh_from_radii_xparam(param, r_arr, axis_c, n_theta=int(n_theta))
        fig3d.add_trace(go.Surface(x=Xo, y=Yo, z=Zo, showscale=False, opacity=0.85))
        fig3d.add_trace(go.Surface(x=Xi, y=Yi, z=Zi, showscale=False, opacity=0.55))
        fig3d.add_trace(go.Scatter3d(x=[a, b], y=[axis_c, axis_c], z=[0, 0], mode="lines", showlegend=False))

        xs_s = np.linspace(a, b, int(n_slices))
        fx_s, gx_s = f_num(xs_s), g_num(xs_s)
        R_s, r_s = radii_from_two_curves_about_line(fx_s, gx_s, axis_c)
        for tr in washer_slice_circles_x(xs_s, R_s, r_s, axis_c):
            fig3d.add_trace(tr)

    elif mode.startswith("Washers: Vertical"):
        R_arr, r_arr = radii_from_two_curves_about_line(v1, v2, axis_c)
        Xo, Yo, Zo = surface_mesh_from_radii_yparam(param, R_arr, axis_c, n_theta=int(n_theta))
        Xi, Yi, Zi = surface_mesh_from_radii_yparam(param, r_arr, axis_c, n_theta=int(n_theta))
        fig3d.add_trace(go.Surface(x=Xo, y=Yo, z=Zo, showscale=False, opacity=0.85))
        fig3d.add_trace(go.Surface(x=Xi, y=Yi, z=Zi, showscale=False, opacity=0.55))
        fig3d.add_trace(go.Scatter3d(x=[axis_c, axis_c], y=[a, b], z=[0, 0], mode="lines", showlegend=False))

        ys_s = np.linspace(a, b, int(n_slices))
        fx_s, gx_s = f_num(ys_s), g_num(ys_s)
        R_s, r_s = radii_from_two_curves_about_line(fx_s, gx_s, axis_c)
        for tr in washer_slice_circles_y(ys_s, R_s, r_s, axis_c):
            fig3d.add_trace(tr)

    elif mode.startswith("Shells:  Vertical"):
        # revolve top/bottom surfaces around x=c for intuition
        thetas = np.linspace(0, 2*np.pi, int(n_theta))
        TH = np.tile(thetas.reshape(-1, 1), (1, param.size))
        RAD = np.abs(param - axis_c)
        RAD2D = np.tile(RAD, (thetas.size, 1))
        Xbase = axis_c + RAD2D*np.cos(TH)
        Zbase = RAD2D*np.sin(TH)
        Ytop = np.tile(v1, (thetas.size, 1))
        Ybot = np.tile(v2, (thetas.size, 1))

        fig3d.add_trace(go.Surface(x=Xbase, y=Ytop, z=Zbase, showscale=False, opacity=0.75))
        fig3d.add_trace(go.Surface(x=Xbase, y=Ybot, z=Zbase, showscale=False, opacity=0.55))
        fig3d.add_trace(go.Scatter3d(x=[axis_c, axis_c],
                                     y=[min(v1.min(), v2.min()), max(v1.max(), v2.max())],
                                     z=[0, 0], mode="lines", showlegend=False))

        xs_s = np.linspace(a, b, int(n_slices))
        for tr in shell_preview_circles_about_xc(xs_s, f_num, g_num, axis_c):
            fig3d.add_trace(tr)

    else:
        # Shells about horizontal axis y=c using dy, with x=f(y), x=g(y)
        # Visualize by revolving the "left/right" boundaries around y=c:
        thetas = np.linspace(0, 2*np.pi, int(n_theta))
        TH = np.tile(thetas.reshape(-1, 1), (1, param.size))
        RAD = np.abs(param - axis_c)          # radius in y
        RAD2D = np.tile(RAD, (thetas.size, 1))

        # two surfaces: x=f(y) and x=g(y), rotated around y=c
        X1 = np.tile(v1, (thetas.size, 1))
        X2 = np.tile(v2, (thetas.size, 1))
        Ybase = axis_c + RAD2D*np.cos(TH)
        Zbase = RAD2D*np.sin(TH)

        fig3d.add_trace(go.Surface(x=X1, y=Ybase, z=Zbase, showscale=False, opacity=0.75))
        fig3d.add_trace(go.Surface(x=X2, y=Ybase, z=Zbase, showscale=False, opacity=0.55))

        fig3d.add_trace(go.Scatter3d(
            x=[min(v1.min(), v2.min()), max(v1.max(), v2.max())],
            y=[axis_c, axis_c], z=[0, 0],
            mode="lines", showlegend=False
        ))

        ys_s = np.linspace(a, b, int(n_slices))
        for tr in shell_preview_circles_about_yc(ys_s, f_num, g_num, axis_c):
            fig3d.add_trace(tr)

    fig3d.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False
    )
    st.plotly_chart(fig3d, use_container_width=True)

st.divider()
st.markdown(
    """
### Notes
- **Shells about x=c (dx)** expects **y=f(x), y=g(x)**.
- **Shells about y=c (dy)** expects **x=f(y), x=g(y)**.
- The 3D view for shells is an *intuition visualization* (it shows rotated boundary surfaces + shell preview circles).
  The **volume computation is the main “truth”**.
"""
)
