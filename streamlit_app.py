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

def safe_parse_function(expr_str: str):
    """
    Parse y = f(x) from a user string into a SymPy expression.
    Supports implicit multiplication: "2x" -> "2*x".
    """
    x = sp.Symbol("x", real=True)
    local_dict = {
        "x": x,
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
    expr = parse_expr(expr_str, local_dict=local_dict, transformations=TRANSFORMS, evaluate=True)
    return x, sp.simplify(expr)

def make_numeric_f(xsym, fsym):
    """
    Turn SymPy f(x) into a numpy-callable function.
    """
    f_num = sp.lambdify(xsym, fsym, modules=["numpy"])
    return f_num

def compute_volume_numeric(f_num, a, b, axis_y):
    """
    Washer/disk volume about horizontal line y = axis_y:
      V = pi * integral_a^b (|f(x) - axis_y|)^2 dx
    """
    def integrand(x):
        r = np.abs(f_num(x) - axis_y)
        return np.pi * (r ** 2)

    val, err = quad(integrand, a, b, limit=200)
    return val, err

def try_symbolic_volume(xsym, fsym, a, b, axis_y):
    """
    Attempt symbolic integral of pi*(f(x)-axis_y)^2 dx on [a,b].
    Note: Abs() makes symbolic integration harder; we integrate (f-axis)^2,
    which is correct when f(x)-axis doesn't change sign, but still yields the
    same radius-squared regardless of sign. So it is safe.
    """
    integrand = sp.pi * (fsym - axis_y) ** 2
    try:
        antideriv = sp.integrate(integrand, xsym)
        exact = sp.simplify(antideriv.subs(xsym, b) - antideriv.subs(xsym, a))
        if exact.has(sp.Integral):
            return None
        return exact
    except Exception:
        return None

def solid_surface_mesh(f_num, a, b, axis_y, n_x=180, n_theta=180):
    """
    Build surface of revolution about y=axis_y by revolving y=f(x) around the x-axis shifted to y=axis_y:
      radius r(x) = |f(x)-axis_y|
      y = axis_y + r cos(theta)
      z = r sin(theta)
      x = x
    """
    xs = np.linspace(a, b, n_x)
    thetas = np.linspace(0, 2*np.pi, n_theta)

    # Ensure 2D grids
    X = np.tile(xs, (n_theta, 1))
    TH = np.tile(thetas.reshape(-1, 1), (1, n_x))

    fx = f_num(xs)
    r = np.abs(fx - axis_y)
    R = np.tile(r, (n_theta, 1))

    Y = axis_y + R * np.cos(TH)
    Z = R * np.sin(TH)
    return X, Y, Z, xs, fx, r

def washer_slices_traces(f_num, a, b, axis_y, n_slices=10):
    """
    Create a set of "washer circles" at sample x locations to illustrate the method.
    """
    xs = np.linspace(a, b, n_slices)
    theta = np.linspace(0, 2*np.pi, 160)

    traces = []
    for xi in xs:
        ri = float(np.abs(f_num(xi) - axis_y))
        y = axis_y + ri * np.cos(theta)
        z = ri * np.sin(theta)
        x = np.full_like(theta, xi)
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=f"x={xi:.3g}", showlegend=False))
    return traces

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Solid of Revolution (Washer Method)", layout="wide")
st.title("Solid of Revolution (Washer/Disk Method, about a horizontal axis)")

with st.sidebar:
    st.header("Inputs")
    expr_str = st.text_input("Function  y = f(x)", value="sqrt(x)")
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("a", value=0.0)
    with col2:
        b = st.number_input("b", value=4.0)

    axis_mode = st.selectbox("Axis of rotation", ["x-axis (y = 0)", "horizontal line (y = c)"])
    if axis_mode == "x-axis (y = 0)":
        axis_y = 0.0
    else:
        axis_y = st.number_input("c (rotate around y = c)", value=0.0)

    st.divider()
    st.subheader("Plot quality")
    n_x = st.slider("Surface resolution (x samples)", 60, 320, 180, 10)
    n_theta = st.slider("Surface resolution (theta samples)", 60, 320, 180, 10)
    n_slices = st.slider("Washer slice preview (count)", 3, 30, 10, 1)

    st.divider()
    st.caption("Tip: use Python/SymPy-style input, e.g. sin(x), exp(-x**2), sqrt(x), (x-1)**2.")

# Validate interval
if b == a:
    st.error("Please choose an interval with b ≠ a.")
    st.stop()
if b < a:
    a, b = b, a
    st.warning("Swapped endpoints so that a < b.")

# Parse and compute
try:
    xsym, fsym = safe_parse_function(expr_str)
    f_num = make_numeric_f(xsym, fsym)

    # Quick sanity check
    test_x = np.linspace(a, b, 5)
    test_y = f_num(test_x)
    if not np.all(np.isfinite(test_y)):
        st.error("Your function produced non-finite values on the interval. Try adjusting the interval or function.")
        st.stop()

except Exception as e:
    st.error(f"Could not parse the function. Details: {e}")
    st.stop()

# Volume
vol_num, vol_err = compute_volume_numeric(f_num, a, b, axis_y)
vol_sym = try_symbolic_volume(xsym, fsym, a, b, axis_y)

# Layout
left, right = st.columns([1, 1])

with left:
    st.subheader("2D view (function + axis)")

    xs = np.linspace(a, b, 600)
    ys = f_num(xs)

    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="y=f(x)"))
    fig2d.add_trace(go.Scatter(x=[a, b], y=[axis_y, axis_y], mode="lines", name="axis y=c"))

    # Fill region between curve and axis (visual aid)
    fig2d.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([ys, np.full_like(xs, axis_y)[::-1]]),
        fill="toself",
        mode="lines",
        name="region (to rotate)",
        opacity=0.2
    ))

    fig2d.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig2d, use_container_width=True)

    st.subheader("Volume")
    st.latex(r"V = \pi \int_a^b \left(|f(x)-c|\right)^2\,dx")
    st.write(f"**Numeric volume:** {vol_num:.10g}")
    st.caption(f"Estimated numeric integration error: ±{vol_err:.2g}")

    if vol_sym is not None:
        st.write("**Symbolic (exact) attempt:**")
        st.latex(sp.latex(sp.simplify(vol_sym)))
    else:
        st.caption("Symbolic integration wasn’t available for this input (numeric result above is reliable).")

with right:
    st.subheader("3D solid of revolution")

    X, Y, Z, xs_surf, fx_surf, r_surf = solid_surface_mesh(
        f_num, a, b, axis_y, n_x=int(n_x), n_theta=int(n_theta)
    )

    fig3d = go.Figure()

    fig3d.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.85, name="surface"))

    # Axis line (x direction at y=axis_y, z=0)
    fig3d.add_trace(go.Scatter3d(
        x=[a, b],
        y=[axis_y, axis_y],
        z=[0, 0],
        mode="lines",
        name="axis",
        showlegend=False
    ))

    # Washer slice preview circles
    for tr in washer_slices_traces(f_num, a, b, axis_y, n_slices=int(n_slices)):
        fig3d.add_trace(tr)

    fig3d.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False
    )

    st.plotly_chart(fig3d, use_container_width=True)

st.divider()
st.markdown(
    """
### Notes / limitations (current version)
- This is the **disk/washer method about a horizontal line** \(y=c\) using **dx**:
  \\,\\(V=\\pi\\int_a^b (\\text{radius})^2\\,dx\\).
- If you want **true washers with a hole** (region between **two** curves \(y=f(x)\) and \(y=g(x)\)),
  tell me and I’ll add a second function input so it computes \\(\\pi\\int (R^2-r^2)\\,dx\\).
- If you want rotation about a **vertical axis** (like \(x=c\)) using washers (dy),
  I can add an option for entering \(x\) as a function of \(y\) (or an implicit solver).
"""
)
