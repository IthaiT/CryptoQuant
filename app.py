"""
Monte Carlo Personal Financial Planning Simulator – v3 (State-Dependent)
=========================================================================
Major upgrade over v2 with state-dependent, discrete-jump dynamics:

  NEW in v3:
  ──────────
  • Job State Machine – 4 states:
        A) Stay/Stagnation  (slow ≈ inflation-only growth)
        B) Job Hop           (instant salary jump, transient 1-month state)
        C) Unemployed        (income ≈ 0 or subsistence)
        D) Survival/Barista  (gig-economy floor wage)
  • Dynamic Expense Elasticity
        – Survival Mode: expenses auto-cut when unemployed / survival
        – Senior Decay: post-senior-age spending declines; medical floor
  • Deficit Financing Strategy
        – Deficits first borrow soft debt (family loans) up to a limit
        – Only overflow goes to high-interest debt (credit cards)
  • Parameter Persistence
        – Save / Load all sidebar parameters to a JSON file

  Retained from v2:
  ─────────────────
  • Income saturation (cap + S-curve deceleration)
  • Lifestyle creep (expenses auto-grow with income)
  • Career peak → income growth decay after peak age
  • Mandatory soft-debt amortisation schedule
  • Emergency fund buffer in the cash-flow waterfall
  • Investment returns with regime-switching (normal + crisis fat-tail)
  • One-off events (wedding, car, house) & personal black-swan risks

Tech: Python 3.9+ | Streamlit | NumPy | Pandas | Plotly
"""

import json
import importlib
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

find_contours = None
try:
    _skimage_measure = importlib.import_module("skimage.measure")
    find_contours = _skimage_measure.find_contours
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="蒙特卡洛财务规划器 v3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Job State Constants
# ──────────────────────────────────────────────────────────────────────
JOB_STAY = 0        # A: 苟着 – stagnation (slow growth)
JOB_HOP = 1         # B: 跳槽 – job hop (transient, salary jump)
JOB_UNEMPLOYED = 2  # C: 失业 – no real income
JOB_SURVIVAL = 3    # D: 兜底 – gig / barista / delivery

JOB_STATE_LABELS = {
    JOB_STAY: "Stay",
    JOB_HOP: "Job Hop",
    JOB_UNEMPLOYED: "Unemployed",
    JOB_SURVIVAL: "Survival",
}

# ──────────────────────────────────────────────────────────────────────
# Parameter Persistence
# ──────────────────────────────────────────────────────────────────────
PARAMS_FILE = Path(__file__).parent / "saved_params.json"

# Every sidebar widget key (used for save / load)
SAVEABLE_KEYS: list = [
    # simulation
    "w_n_simulations", "w_n_years",
    # initial state
    "w_initial_cash", "w_high_debt_init", "w_low_debt_init",
    # income
    "w_monthly_income", "w_income_cap",
    "w_stay_growth_mean_pct", "w_stay_growth_vol_pct",
    # career
    "w_current_age", "w_career_peak_age", "w_post_peak_decay",
    # job state machine
    "w_hop_annual_prob_pct", "w_hop_jump_mean_pct", "w_hop_jump_vol_pct",
    "w_layoff_annual_prob_pct",
    "w_unemp_income_pct", "w_reemploy_monthly_prob_pct",
    "w_to_survival_monthly_prob_pct", "w_survival_wage",
    "w_recovery_monthly_prob_pct", "w_reemploy_haircut_pct",
    # expenses
    "w_monthly_expense", "w_annual_inflation_pct",
    "w_lifestyle_creep", "w_expense_cap",
    "w_survival_expense_ratio",
    "w_senior_age", "w_senior_decay_pct", "w_medical_floor",
    # debt
    "w_high_debt_apr_pct", "w_low_debt_apr_pct",
    "w_annual_soft_repay", "w_soft_debt_limit",
    # investments
    "w_invest_return_mean_pct", "w_invest_return_vol_pct",
    "w_dca_start_year", "w_dca_surplus_ratio",
    "w_crisis_annual_prob_pct", "w_crisis_drawdown_mean_pct",
    "w_crisis_drawdown_vol_pct",
    # emergency fund
    "w_emergency_fund_months",
    # one-off events
    "w_wedding_toggle", "w_wedding_year", "w_wedding_cost",
    "w_car_toggle", "w_car_year", "w_car_cost",
    "w_house_toggle", "w_house_year", "w_house_down",
    "w_mortgage_monthly", "w_rent_in_expenses",
    "w_black_swan_prob_pct", "w_black_swan_cost_min", "w_black_swan_cost_max",
]


def _load_saved_params() -> dict:
    """Read saved widget values from disk (best-effort)."""
    try:
        if PARAMS_FILE.exists():
            with open(PARAMS_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def _save_current_params():
    """Persist current sidebar widget values to JSON."""
    data = {}
    for k in SAVEABLE_KEYS:
        if k in st.session_state:
            v = st.session_state[k]
            # numpy types → native Python for JSON serialisation
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            data[k] = v
    with open(PARAMS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# Inject saved values into session_state ONCE at app startup
if "_params_loaded" not in st.session_state:
    _saved = _load_saved_params()
    for _k, _v in _saved.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v
    st.session_state["_params_loaded"] = True


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR – Every tuneable parameter lives here
# ══════════════════════════════════════════════════════════════════════


def build_sidebar_inputs() -> dict:
    """Build all user-adjustable parameters; return a processed dict."""

    sb = st.sidebar
    sb.title("⚙️ 模拟参数")

    # ── Save / Load ──────────────────────────────────────────────────
    sb.header("💾 参数持久化")
    col_s, col_l = sb.columns(2)
    with col_s:
        if st.button("💾 保存", use_container_width=True,
                  help="将当前参数保存到磁盘"):
            _save_current_params()
            st.toast("✅ 参数已保存！", icon="💾")
    with col_l:
        if st.button("📂 读取", use_container_width=True,
                  help="从磁盘读取上次保存的参数"):
            loaded = _load_saved_params()
            if loaded:
                for k, v in loaded.items():
                    st.session_state[k] = v
                st.toast("✅ 参数已加载！", icon="📂")
                st.rerun()
            else:
                st.toast("⚠️ 未找到已保存参数。", icon="⚠️")

    # ── Simulation Settings ──────────────────────────────────────────
    sb.header("🎲 模拟设置")
    n_simulations = sb.slider(
        "蒙特卡洛运行次数", 100, 10_000, 1_000, step=100,
        key="w_n_simulations",
        help="次数越多，概率带更平滑，但运行更慢。",
    )
    n_years = sb.slider(
        "预测期（年）", 5, 50, 20,
        key="w_n_years",
    )

    # ── Initial State ────────────────────────────────────────────────
    sb.header("💰 初始状态")
    initial_cash = sb.number_input(
        "初始现金 / 储蓄", 0, 10_000_000, 20_000, step=1_000,
        key="w_initial_cash",
        help="t=0 时的流动资金（也可作为应急金种子）。",
    )
    high_debt_init = sb.number_input(
        "高息负债", 0, 50_000_000, 250_000, step=10_000,
        key="w_high_debt_init",
    )
    low_debt_init = sb.number_input(
        "软负债 / 低息负债（如家人借款）", 0, 50_000_000,
        650_000, step=10_000,
        key="w_low_debt_init",
    )

    # ── Income & Saturation ──────────────────────────────────────────
    sb.header("📈 收入与上限")
    monthly_income = sb.number_input(
        "家庭月净收入", 0, 10_000_000, 28_000, step=1_000,
        key="w_monthly_income",
    )
    income_cap = sb.number_input(
        "月收入上限（工资天花板）", 0, 10_000_000, 80_000,
        step=5_000, key="w_income_cap",
        help="当收入接近该上限时，增长会逐步放缓（Logistic 饱和）。",
    )

    sb.subheader("📊 在职 / 停滞增长")
    stay_growth_mean_pct = sb.slider(
        "在职状态年化增长均值（%）", 0.0, 15.0, 3.0, 0.5,
        key="w_stay_growth_mean_pct",
        help="留在当前岗位时的慢速增长（约等于通胀）。",
    )
    stay_growth_vol_pct = sb.slider(
        "在职状态年化增长波动（%）", 0.0, 10.0, 2.0, 0.5,
        key="w_stay_growth_vol_pct",
    )

    # ── Career Curve ─────────────────────────────────────────────────
    sb.subheader("👔 职业曲线")
    current_age = sb.number_input(
        "当前年龄", 18, 70, 28, key="w_current_age",
    )
    career_peak_age = sb.number_input(
        "职业峰值年龄", 30, 70, 45, key="w_career_peak_age",
        help="超过该年龄后，年化收入增长线性衰减，并可能转负（职业回落/退休）。",
    )
    post_peak_decay = sb.slider(
        "峰值后增长衰减率", 0.0, 1.0, 0.15, 0.01,
        key="w_post_peak_decay",
        help="超过峰值后的每一年，增长均值按（超过年数 × 衰减率）下调。",
    )

    # ── Job State Machine ────────────────────────────────────────────
    sb.header("🔄 职业状态机")

    sb.subheader("🚀 跳槽（状态 B）")
    hop_annual_prob_pct = sb.slider(
        "年化跳槽概率（%）", 0.0, 50.0, 10.0, 1.0,
        key="w_hop_annual_prob_pct",
        help="每年发生跳槽并伴随薪资跃迁的概率。",
    )
    hop_jump_mean_pct = sb.slider(
        "跳槽薪资跃升均值（%）", 0.0, 80.0, 25.0, 1.0,
        key="w_hop_jump_mean_pct",
        help="发生跳槽时的平均涨薪幅度。",
    )
    hop_jump_vol_pct = sb.slider(
        "跳槽薪资跃升波动（%）", 0.0, 30.0, 10.0, 1.0,
        key="w_hop_jump_vol_pct",
    )

    sb.subheader("😰 失业（状态 C）")
    layoff_annual_prob_pct = sb.slider(
        "年化失业/裁员概率（%）", 0.0, 50.0, 10.0, 1.0,
        key="w_layoff_annual_prob_pct",
    )
    unemp_income_pct = sb.slider(
        "失业期间收入（占原收入%）", 0, 100, 20, 5,
        key="w_unemp_income_pct",
        help="例如社保/补偿等，按原收入比例计。",
    )
    reemploy_monthly_prob_pct = sb.slider(
        "月度再就业概率（%）", 0.0, 50.0, 15.0, 1.0,
        key="w_reemploy_monthly_prob_pct",
        help="失业时每月重新找到工作的概率。",
    )
    to_survival_monthly_prob_pct = sb.slider(
        "月度转入兜底模式概率（%）", 0.0, 20.0, 3.0, 0.5,
        key="w_to_survival_monthly_prob_pct",
        help="长期失业后，每月转入兜底/打零工模式的概率。",
    )

    sb.subheader("🛟 兜底模式（状态 D）")
    survival_wage = sb.number_input(
        "兜底收入下限（月）", 0, 50_000, 6_000, step=500,
        key="w_survival_wage",
        help="最低月收入（如跑单、配送等）。",
    )
    recovery_monthly_prob_pct = sb.slider(
        "月度恢复至就业概率（%）", 0.0, 30.0, 5.0, 1.0,
        key="w_recovery_monthly_prob_pct",
        help="每月从兜底模式恢复到正常就业的概率。",
    )

    sb.subheader("🔄 再就业")
    reemploy_haircut_pct = sb.slider(
        "再就业薪资折损（%）", 0, 50, 10, 1,
        key="w_reemploy_haircut_pct",
        help="失业/兜底后重新就业时的薪资折扣。",
    )

    # ── Expenses, Inflation & Lifestyle Creep ────────────────────────
    sb.header("🛒 支出与生活方式")
    monthly_expense = sb.number_input(
        "基础月支出", 0, 10_000_000, 13_000, step=1_000,
        key="w_monthly_expense",
    )
    annual_inflation_pct = sb.slider(
        "年化通胀率（%）", 0.0, 15.0, 3.0, 0.5,
        key="w_annual_inflation_pct",
    )
    lifestyle_creep = sb.slider(
        "生活方式膨胀系数（0–1）", 0.0, 1.0, 0.20, 0.05,
        key="w_lifestyle_creep",
        help="当月收入增加 ΔI 时，支出自动增加 ΔI × 系数。",
    )
    expense_cap = sb.number_input(
        "月支出上限", 0, 10_000_000, 40_000, step=5_000,
        key="w_expense_cap",
        help="生活性支出不会超过该上限。",
    )

    sb.subheader("🧘 动态支出弹性")
    survival_expense_ratio = sb.slider(
        "失业/兜底支出比例", 0.3, 1.0, 0.60, 0.05,
        key="w_survival_expense_ratio",
        help="失业或兜底模式时，支出会自动降至基础支出的该比例。",
    )
    senior_age = sb.number_input(
        "高龄消费衰减起始年龄", 50, 80, 60, key="w_senior_age",
        help="超过该年龄后，支出按年衰减。",
    )
    senior_decay_pct = sb.slider(
        "高龄年化支出衰减（%）", 0.0, 5.0, 1.0, 0.5,
        key="w_senior_decay_pct",
        help="超过起始年龄后，每年按该比例下降。",
    )
    medical_floor = sb.number_input(
        "医疗刚性支出下限（月）", 0, 100_000, 5_000, step=500,
        key="w_medical_floor",
        help="支出不会低于该下限（医疗/生存刚需）。",
    )

    # ── Debt & Deficit Financing ─────────────────────────────────────
    sb.header("🏦 债务与赤字融资")
    high_debt_apr_pct = sb.slider(
        "高息债 APR（%）", 0.0, 50.0, 20.0, 0.5,
        key="w_high_debt_apr_pct",
    )
    low_debt_apr_pct = sb.slider(
        "低息债 APR（%）", 0.0, 20.0, 0.0, 0.5,
        key="w_low_debt_apr_pct",
    )
    annual_soft_repay = sb.number_input(
        "软债年强制还款额", 0, 10_000_000, 60_000,
        step=5_000, key="w_annual_soft_repay",
        help="软债固定年摊还（如 6 万/年=5 千/月），在高息债还款之后执行。",
    )

    sb.subheader("💸 赤字融资策略")
    soft_debt_limit = sb.number_input(
        "软债额度上限（亲友可借总额）", 0, 50_000_000,
        200_000, step=10_000, key="w_soft_debt_limit",
        help="现金流为负时优先借软债（低息），超出该上限再转高息债。",
    )

    # ── Investment Returns ───────────────────────────────────────────
    sb.header("📊 投资收益")
    invest_return_mean_pct = sb.slider(
        "投资年化收益均值（%）", -10.0, 30.0, 8.0, 0.5,
        key="w_invest_return_mean_pct",
    )
    invest_return_vol_pct = sb.slider(
        "投资收益波动率（%）", 0.0, 40.0, 15.0, 0.5,
        key="w_invest_return_vol_pct",
    )

    sb.subheader("🧺 定投设置")
    dca_start_year = sb.number_input(
        "定投起始年份（从现在起）", 0, 50, 0,
        key="w_dca_start_year",
        help="从该年份开始定投，0 表示立即开始。",
    )
    dca_surplus_ratio = sb.slider(
        "定投比例（收入-开销）", 0.0, 1.0, 1.0, 0.05,
        key="w_dca_surplus_ratio",
        help="月定投目标 = 比例 × max(收入-开销, 0)。",
    )

    sb.subheader("💥 肥尾危机风险")
    crisis_annual_prob_pct = sb.slider(
        "金融危机年化概率（%）", 0.0, 30.0, 8.0, 1.0,
        key="w_crisis_annual_prob_pct",
        help="每年进入深度回撤状态的概率。8% 约等于 12 年一次。",
    )
    crisis_drawdown_mean_pct = sb.slider(
        "危机期回撤均值（%）", -70.0, -10.0, -35.0, 5.0,
        key="w_crisis_drawdown_mean_pct",
        help="危机年份的平均年化收益。",
    )
    crisis_drawdown_vol_pct = sb.slider(
        "危机期回撤波动（%）", 5.0, 30.0, 10.0, 1.0,
        key="w_crisis_drawdown_vol_pct",
    )

    # ── Emergency Fund ───────────────────────────────────────────────
    sb.header("🛡️ 应急金")
    emergency_fund_months = sb.slider(
        "应急金目标（月支出倍数）", 0, 12, 6,
        key="w_emergency_fund_months",
        help="在进入投资前，优先把现金补足到该月数的支出规模。",
    )

    # ── Large One-Off Events ─────────────────────────────────────────
    sb.header("🎉 大额一次性事件")

    wedding_toggle = sb.checkbox(
        "计划婚礼", value=False, key="w_wedding_toggle",
    )
    wedding_year = sb.number_input(
        "婚礼 - 距今第几年", 1, 40, 2,
        disabled=not wedding_toggle, key="w_wedding_year",
    )
    wedding_cost = sb.number_input(
        "婚礼 - 费用", 0, 10_000_000, 100_000, step=10_000,
        disabled=not wedding_toggle, key="w_wedding_cost",
    )

    car_toggle = sb.checkbox(
        "购车", value=False, key="w_car_toggle",
    )
    car_year = sb.number_input(
        "购车 - 距今第几年", 1, 40, 3,
        disabled=not car_toggle, key="w_car_year",
    )
    car_cost = sb.number_input(
        "购车 - 费用", 0, 10_000_000, 150_000, step=10_000,
        disabled=not car_toggle, key="w_car_cost",
    )

    house_toggle = sb.checkbox(
        "购房（首付）", value=False, key="w_house_toggle",
    )
    house_year = sb.number_input(
        "购房 - 距今第几年", 1, 40, 5,
        disabled=not house_toggle, key="w_house_year",
    )
    house_down = sb.number_input(
        "购房 - 首付", 0, 50_000_000, 500_000, step=50_000,
        disabled=not house_toggle, key="w_house_down",
    )
    mortgage_monthly = sb.number_input(
        "房贷月供", 0, 500_000, 8_000, step=500,
        disabled=not house_toggle, key="w_mortgage_monthly",
    )
    rent_in_expenses = sb.number_input(
        "当前支出中的房租", 0, 500_000, 5_000, step=500,
        disabled=not house_toggle, key="w_rent_in_expenses",
    )

    sb.subheader("🦢 个人黑天鹅事件")
    black_swan_prob_pct = sb.slider(
        "黑天鹅年化概率（%）", 0.0, 20.0, 2.0, 0.5,
        key="w_black_swan_prob_pct",
    )
    black_swan_cost_min = sb.number_input(
        "黑天鹅损失 - 最小", 0, 50_000_000, 50_000, step=10_000,
        key="w_black_swan_cost_min",
    )
    black_swan_cost_max = sb.number_input(
        "黑天鹅损失 - 最大", 0, 50_000_000, 300_000, step=10_000,
        key="w_black_swan_cost_max",
    )

    # ──────────────────────────────────────────────────────────────────
    # Pack everything into a processed params dict
    # ──────────────────────────────────────────────────────────────────
    return dict(
        # simulation
        n_simulations=n_simulations,
        n_years=n_years,
        n_months=n_years * 12,
        # initial state
        initial_cash=float(initial_cash),
        high_debt_init=float(high_debt_init),
        low_debt_init=float(low_debt_init),
        # income
        monthly_income=float(monthly_income),
        income_cap=float(max(income_cap, 1)),
        stay_growth_mean=stay_growth_mean_pct / 100.0,
        stay_growth_vol=stay_growth_vol_pct / 100.0,
        # career
        current_age=current_age,
        career_peak_age=career_peak_age,
        post_peak_decay=post_peak_decay,
        # job state machine
        hop_annual_prob=hop_annual_prob_pct / 100.0,
        hop_jump_mean=hop_jump_mean_pct / 100.0,
        hop_jump_vol=hop_jump_vol_pct / 100.0,
        layoff_annual_prob=layoff_annual_prob_pct / 100.0,
        unemp_income_pct=unemp_income_pct / 100.0,
        reemploy_monthly_prob=reemploy_monthly_prob_pct / 100.0,
        to_survival_monthly_prob=to_survival_monthly_prob_pct / 100.0,
        survival_wage=float(survival_wage),
        recovery_monthly_prob=recovery_monthly_prob_pct / 100.0,
        reemploy_haircut=reemploy_haircut_pct / 100.0,
        # expenses
        monthly_expense=float(monthly_expense),
        annual_inflation=annual_inflation_pct / 100.0,
        lifestyle_creep=lifestyle_creep,
        expense_cap=float(max(expense_cap, 1)),
        survival_expense_ratio=survival_expense_ratio,
        senior_age=senior_age,
        senior_decay=senior_decay_pct / 100.0,
        medical_floor=float(medical_floor),
        # debt
        high_debt_apr=high_debt_apr_pct / 100.0,
        low_debt_apr=low_debt_apr_pct / 100.0,
        annual_soft_repay=float(annual_soft_repay),
        monthly_soft_repay=float(annual_soft_repay) / 12.0,
        soft_debt_limit=float(soft_debt_limit),
        # investments
        invest_return_mean=invest_return_mean_pct / 100.0,
        invest_return_vol=invest_return_vol_pct / 100.0,
        dca_start_month=int(dca_start_year) * 12,
        dca_surplus_ratio=float(dca_surplus_ratio),
        # NOTE: v2 had a bug – crisis_drawdown_vol was divided by 100 twice.
        #       Fixed in v3: each slider already converts % → fraction once.
        crisis_annual_prob=crisis_annual_prob_pct / 100.0,
        crisis_drawdown_mean=crisis_drawdown_mean_pct / 100.0,
        crisis_drawdown_vol=crisis_drawdown_vol_pct / 100.0,
        # emergency fund
        emergency_fund_months=emergency_fund_months,
        # one-off events
        wedding_toggle=wedding_toggle,
        wedding_month=wedding_year * 12 if wedding_toggle else -1,
        wedding_cost=float(wedding_cost) if wedding_toggle else 0.0,
        car_toggle=car_toggle,
        car_month=car_year * 12 if car_toggle else -1,
        car_cost=float(car_cost) if car_toggle else 0.0,
        house_toggle=house_toggle,
        house_month=house_year * 12 if house_toggle else -1,
        house_down=float(house_down) if house_toggle else 0.0,
        mortgage_monthly=float(mortgage_monthly) if house_toggle else 0.0,
        rent_in_expenses=float(rent_in_expenses) if house_toggle else 0.0,
        black_swan_prob=black_swan_prob_pct / 100.0,
        black_swan_cost_min=float(black_swan_cost_min),
        black_swan_cost_max=float(black_swan_cost_max),
    )


# ══════════════════════════════════════════════════════════════════════
# CORE SIMULATION ENGINE  –  v3
# ══════════════════════════════════════════════════════════════════════
#
# Design principles
# -----------------
#   • Vectorised over N simulations (columns), sequential over T months.
#   • **Job State Machine** (Markov chain per sim):
#       STAY ──(p_hop)──→ HOP ───→ STAY   (transient salary jump)
#       STAY ──(p_layoff)──→ UNEMPLOYED
#       UNEMPLOYED ──(p_reemploy)──→ STAY
#       UNEMPLOYED ──(p_to_surv)──→ SURVIVAL
#       SURVIVAL ──(p_recovery)──→ STAY
#   • **Dynamic Expense Elasticity**:
#       – Unemployed / Survival → expenses × survival_ratio
#       – Senior age → expenses decay annually; medical floor
#   • Strict *waterfall* for monthly surplus allocation:
#       1) Living Expenses (already deducted to derive gross surplus)
#       2) High-Interest Debt principal
#       3) Mandatory Soft-Debt Amortisation instalment
#       4) Emergency Fund up to target
#       5) Investment / DCA
#   • **Deficit Financing** (reverse waterfall):
#       Draw Cash → Liquidate Investments → Borrow Soft Debt (up to
#       limit) → Accrue High-Interest Debt (death spiral)
#   • Income growth features:
#       – Logistic saturation toward *income_cap*
#       – Career-peak decay (post-peak, growth mean shrinks)
#       – Lifestyle creep: positive income Δ feeds back into expenses
#   • Investment returns use *regime-switching*:
#       Each sim-year is "normal" or "crisis" (Bernoulli).
# ══════════════════════════════════════════════════════════════════════


def run_simulation(p: dict) -> dict:
    """
    Run the full Monte Carlo simulation.

    Returns
    -------
    dict with arrays shaped (n_months+1, n_simulations):
        cash, high_debt, low_debt, investments,
        income_ts, expense_ts, net_flow_ts, emergency_target,
        job_state_ts
    """
    N = p["n_simulations"]
    T = p["n_months"]
    rng = np.random.default_rng()

    # ── Output arrays (T+1 × N) ─────────────────────────────────────
    cash         = np.zeros((T + 1, N))
    high_debt    = np.zeros((T + 1, N))
    low_debt     = np.zeros((T + 1, N))
    investments  = np.zeros((T + 1, N))
    investment_principal = np.zeros((T + 1, N))
    income_ts    = np.zeros((T + 1, N))
    expense_ts   = np.zeros((T + 1, N))
    base_expense_ts = np.zeros((T + 1, N))
    net_flow_ts  = np.zeros((T + 1, N))
    dca_ts       = np.zeros((T + 1, N))
    emerg_tgt    = np.zeros((T + 1, N))
    job_state_ts = np.zeros((T + 1, N), dtype=np.int8)

    # ── Initialise t = 0 ─────────────────────────────────────────────
    cash[0]      = p["initial_cash"]
    high_debt[0] = p["high_debt_init"]
    low_debt[0]  = p["low_debt_init"]
    # job_state_ts[0] = JOB_STAY (already 0)

    # ── Mutable per-sim state ────────────────────────────────────────
    cur_income        = np.full(N, p["monthly_income"])
    cur_expense       = np.full(N, p["monthly_expense"])
    prev_income       = cur_income.copy()
    pre_layoff_income = cur_income.copy()
    job_state         = np.zeros(N, dtype=np.int8)  # all start STAY
    house_bought      = np.zeros(N, dtype=bool)

    high_mr = p["high_debt_apr"] / 12.0
    low_mr  = p["low_debt_apr"]  / 12.0

    # ── Monthly transition probabilities (from annual where needed) ──
    p_hop_m    = 1.0 - (1.0 - p["hop_annual_prob"]) ** (1.0 / 12.0)
    p_layoff_m = 1.0 - (1.0 - p["layoff_annual_prob"]) ** (1.0 / 12.0)
    p_reemploy = p["reemploy_monthly_prob"]
    p_to_surv  = p["to_survival_monthly_prob"]
    p_recovery = p["recovery_monthly_prob"]

    # ── Pre-draw stochastic inputs ───────────────────────────────────
    n_yr = (T // 12) + 2

    # a) Stay growth shocks (once per year per sim)
    stay_growth_raw = rng.normal(
        p["stay_growth_mean"],
        max(p["stay_growth_vol"], 1e-9),
        size=(n_yr, N),
    )

    # b) Job-hop salary jump draws
    hop_jumps = rng.normal(
        p["hop_jump_mean"],
        max(p["hop_jump_vol"], 1e-9),
        size=(T, N),
    )

    # c) Investment returns – regime switching
    crisis_flags = rng.random((n_yr, N)) < p["crisis_annual_prob"]

    inv_mu_m   = p["invest_return_mean"] / 12.0
    inv_sig_m  = p["invest_return_vol"]  / np.sqrt(12.0)
    inv_normal = rng.normal(inv_mu_m, max(inv_sig_m, 1e-9), size=(T, N))

    crisis_mu_m  = p["crisis_drawdown_mean"] / 12.0
    crisis_sig_m = p["crisis_drawdown_vol"]  / np.sqrt(12.0)
    inv_crisis   = rng.normal(
        crisis_mu_m, max(crisis_sig_m, 1e-9), size=(T, N),
    )

    crisis_mask = np.zeros((T, N), dtype=bool)
    for yr in range(n_yr):
        ms, me = yr * 12, min((yr + 1) * 12, T)
        if ms < T:
            crisis_mask[ms:me, :] = crisis_flags[yr, :]
    invest_shocks = np.where(crisis_mask, inv_crisis, inv_normal)

    # d) State transition draws (one uniform per month per sim)
    state_draws = rng.random((T, N))

    # e) Black swan draws
    bs_draws  = rng.random((T, N))
    bs_m_prob = 1.0 - (1.0 - p["black_swan_prob"]) ** (1.0 / 12.0)
    bs_cost   = rng.uniform(
        p["black_swan_cost_min"],
        max(p["black_swan_cost_max"], p["black_swan_cost_min"] + 1),
        size=(T, N),
    )

    # ══════════════════════════════════════════════════════════════════
    # MAIN MONTHLY LOOP
    # ══════════════════════════════════════════════════════════════════
    for t in range(1, T + 1):
        idx    = t - 1                     # 0-based index
        yr_idx = idx // 12                 # simulation year index

        # Copy previous balances
        c   = cash[t - 1].copy()
        hd  = high_debt[t - 1].copy()
        ld  = low_debt[t - 1].copy()
        inv = investments[t - 1].copy()
        inv_principal = investment_principal[t - 1].copy()

        # ==============================================================
        # A. JOB STATE MACHINE TRANSITIONS
        #
        #    Transitions depend on the PREVIOUS state.  A single uniform
        #    draw per sim disambiguates which transition fires (states
        #    are mutually exclusive so thresholds don't collide).
        # ==============================================================
        prev_state = job_state.copy()
        draws      = state_draws[idx]
        new_state  = job_state.copy()

        # From STAY ─────────────────────────────────────────────────
        stay = (job_state == JOB_STAY)
        new_state[stay & (draws < p_hop_m)] = JOB_HOP
        new_state[stay & (draws >= p_hop_m)
                  & (draws < p_hop_m + p_layoff_m)] = JOB_UNEMPLOYED

        # From HOP → always STAY next month (transient) ────────────
        new_state[job_state == JOB_HOP] = JOB_STAY

        # From UNEMPLOYED → STAY (re-employed) or SURVIVAL ─────────
        unemp = (job_state == JOB_UNEMPLOYED)
        new_state[unemp & (draws < p_reemploy)] = JOB_STAY
        new_state[unemp & (draws >= p_reemploy)
                  & (draws < p_reemploy + p_to_surv)] = JOB_SURVIVAL

        # From SURVIVAL → STAY (recovered) ─────────────────────────
        surv = (job_state == JOB_SURVIVAL)
        new_state[surv & (draws < p_recovery)] = JOB_STAY

        job_state = new_state

        # ── Apply salary effects of transitions ──────────────────────

        # Entering HOP: instant salary jump
        just_hopped = (job_state == JOB_HOP) & (prev_state == JOB_STAY)
        if np.any(just_hopped):
            jump = np.maximum(hop_jumps[idx], -0.30)  # floor: -30 %
            cur_income[just_hopped] *= (1.0 + jump[just_hopped])
            cur_income = np.minimum(cur_income, p["income_cap"])

        # Entering UNEMPLOYED: save pre-layoff income
        just_laid_off = ((job_state == JOB_UNEMPLOYED)
                         & (prev_state != JOB_UNEMPLOYED)
                         & (prev_state != JOB_SURVIVAL))
        pre_layoff_income[just_laid_off] = cur_income[just_laid_off]

        # Re-employed from UNEMPLOYED or SURVIVAL
        re_employed = ((job_state == JOB_STAY)
                       & ((prev_state == JOB_UNEMPLOYED)
                          | (prev_state == JOB_SURVIVAL)))
        if np.any(re_employed):
            cur_income[re_employed] = (
                pre_layoff_income[re_employed]
                * (1.0 - p["reemploy_haircut"])
            )
            # Never below survival wage
            cur_income[re_employed] = np.maximum(
                cur_income[re_employed], p["survival_wage"],
            )

        # ==============================================================
        # B. ANNUAL INCOME GROWTH  (applied at month 1, 13, 25, …)
        #    Only for employed sims (STAY or HOP).
        #    1. Career-peak decay modifies the effective growth mean
        #    2. Logistic saturation dampens growth near income_cap
        #    3. Lifestyle creep feeds back into expenses
        # ==============================================================
        if t % 12 == 1 and t > 1:
            employed = (job_state == JOB_STAY) | (job_state == JOB_HOP)

            # Career-peak modifier
            age_now  = p["current_age"] + t / 12.0
            yrs_past = max(0.0, age_now - p["career_peak_age"])
            career_mod = 1.0 - yrs_past * p["post_peak_decay"]

            raw_g = stay_growth_raw[yr_idx]
            eff_g = career_mod * raw_g

            # Logistic saturation: growth → 0 as income → cap
            cap = p["income_cap"]
            saturation = np.clip(
                1.0 - (cur_income / cap) ** 2, 0.0, 1.0,
            )
            damped_g = eff_g * saturation

            g_factor = np.maximum(1.0 + damped_g, 0.5)
            prev_income[:] = cur_income.copy()

            # Only apply growth to employed sims
            cur_income[employed] *= g_factor[employed]
            cur_income = np.minimum(cur_income, cap)

            # Update pre-layoff reference for future layoffs
            pre_layoff_income = np.maximum(pre_layoff_income, cur_income)

            # Lifestyle creep: only upward Δ creeps into expenses
            delta_i = np.maximum(cur_income - prev_income, 0.0)
            creep   = delta_i * p["lifestyle_creep"]
            cur_expense += creep
            cur_expense  = np.minimum(cur_expense, p["expense_cap"])

        # ==============================================================
        # C. ANNUAL EXPENSE INFLATION + SENIOR DECAY
        # ==============================================================
        if t % 12 == 1 and t > 1:
            cur_expense *= (1.0 + p["annual_inflation"])
            cur_expense  = np.minimum(cur_expense, p["expense_cap"])

            # Senior decay: post-senior-age, expenses shrink annually
            age_now = p["current_age"] + t / 12.0
            if age_now > p["senior_age"]:
                cur_expense *= (1.0 - p["senior_decay"])
                cur_expense  = np.maximum(
                    cur_expense, p["medical_floor"],
                )

        # ==============================================================
        # D. EFFECTIVE INCOME & EXPENSES (state-dependent)
        # ==============================================================
        eff_income = np.zeros(N)
        is_stay_or_hop = (job_state == JOB_STAY) | (job_state == JOB_HOP)
        is_unemp       = (job_state == JOB_UNEMPLOYED)
        is_surv        = (job_state == JOB_SURVIVAL)

        eff_income[is_stay_or_hop] = cur_income[is_stay_or_hop]
        eff_income[is_unemp] = (
            cur_income[is_unemp] * p["unemp_income_pct"]
        )
        eff_income[is_surv] = p["survival_wage"]

        # Dynamic expense elasticity
        eff_expense = cur_expense.copy()
        distressed  = is_unemp | is_surv
        eff_expense[distressed] *= p["survival_expense_ratio"]
        # Universal floor: can't spend less than medical minimum
        eff_expense = np.maximum(eff_expense, p["medical_floor"])

        # Housing swap (mortgage replaces rent after house purchase)
        if p["house_toggle"]:
            eff_expense[house_bought] += (
                p["mortgage_monthly"] - p["rent_in_expenses"]
            )
        eff_expense = np.maximum(eff_expense, 0.0)

        # ==============================================================
        # E. LARGE ONE-OFF EVENTS
        # ==============================================================
        event_cost = np.zeros(N)
        if p["wedding_toggle"] and t == p["wedding_month"]:
            event_cost += p["wedding_cost"]
        if p["car_toggle"] and t == p["car_month"]:
            event_cost += p["car_cost"]
        if p["house_toggle"] and t == p["house_month"]:
            event_cost += p["house_down"]
            house_bought[:] = True
        bs_hit = bs_draws[idx] < bs_m_prob
        event_cost[bs_hit] += bs_cost[idx][bs_hit]

        # ==============================================================
        # F. WATERFALL – strict cash-flow allocation
        #
        #    gross_surplus = income − living_expenses − event_costs
        #
        #    Positive path (have money):
        #      W1  Pay High-Interest Debt principal
        #      W2  Mandatory Soft-Debt Amortisation instalment
        #      W3  Fill Emergency Fund → target
        #      W4  Invest (DCA)
        #
        #    Negative path (deficit financing):
        #      1. Draw Cash
        #      2. Liquidate Investments
        #      3. Borrow Soft Debt up to limit
        #      4. Remaining → High-Interest Debt (death spiral)
        # ==============================================================
        total_out     = eff_expense + event_cost
        gross_surplus = eff_income - total_out

        # Emergency-fund target: N months of CURRENT effective expenses
        ef_target = eff_expense * p["emergency_fund_months"]

        # ── POSITIVE SURPLUS ─────────────────────────────────────────
        pos     = gross_surplus > 0
        surplus = np.where(pos, gross_surplus, 0.0)

        # W1 – High-interest debt principal
        pay_hi   = np.minimum(surplus, hd)
        hd      -= pay_hi
        surplus -= pay_hi

        # W2 – Mandatory soft-debt amortisation (monthly instalment)
        msr      = p["monthly_soft_repay"]
        pay_lo   = np.minimum(surplus, np.minimum(msr, ld))
        ld      -= pay_lo
        surplus -= pay_lo

        # W3 – Fill emergency fund up to target
        ef_gap   = np.maximum(ef_target - c, 0.0)
        fill_ef  = np.minimum(surplus, ef_gap)
        c       += fill_ef
        surplus -= fill_ef

        # W4 – DCA with gating + ratio cap
        #      target_dca = ratio × max(income - expenses, 0)
        #      only active after dca_start_month
        if t >= p["dca_start_month"]:
            base_surplus = np.maximum(eff_income - eff_expense, 0.0)
            target_dca = p["dca_surplus_ratio"] * base_surplus
            dca_amt = np.minimum(surplus, target_dca)
        else:
            dca_amt = np.zeros_like(surplus)

        inv += dca_amt
        inv_principal += dca_amt
        surplus -= dca_amt

        # Any unallocated positive surplus remains in cash
        c += surplus

        # ── NEGATIVE SURPLUS (DEFICIT) – multi-tier financing ────────
        neg     = gross_surplus < 0
        deficit = np.where(neg, -gross_surplus, 0.0)

        # Tier 1: Draw cash (emergency fund)
        draw_c   = np.minimum(deficit, np.maximum(c, 0.0))
        c       -= draw_c
        deficit -= draw_c

        # Tier 2: Liquidate investments
        inv_before_draw = np.maximum(inv, 0.0)
        draw_i   = np.minimum(deficit, inv_before_draw)
        principal_reduction = np.where(
            inv_before_draw > 1e-12,
            draw_i * (inv_principal / inv_before_draw),
            0.0,
        )
        inv_principal = np.maximum(inv_principal - principal_reduction, 0.0)
        inv     -= draw_i
        deficit -= draw_i

        # Tier 3: Borrow soft debt (family / friends) up to limit
        soft_capacity = np.maximum(p["soft_debt_limit"] - ld, 0.0)
        borrow_soft   = np.minimum(deficit, soft_capacity)
        ld            += borrow_soft
        deficit       -= borrow_soft

        # Tier 4: Remaining deficit → high-interest debt (death spiral)
        hd += deficit

        # ==============================================================
        # G. COMPOUND INTEREST (month-end)
        # ==============================================================
        # Investment returns (regime-switching baked into invest_shocks)
        ret = 1.0 + invest_shocks[idx]
        ret = np.maximum(ret, 0.0)
        inv = np.maximum(inv, 0.0) * ret

        # High-interest debt accrual
        hd *= (1.0 + high_mr)

        # Low-interest debt accrual
        if low_mr > 0:
            ld *= (1.0 + low_mr)

        # ==============================================================
        # H. STORE
        # ==============================================================
        cash[t]         = c
        high_debt[t]    = hd
        low_debt[t]     = ld
        investments[t]  = inv
        investment_principal[t] = inv_principal
        income_ts[t]    = eff_income
        expense_ts[t]   = total_out
        base_expense_ts[t] = eff_expense
        net_flow_ts[t]  = gross_surplus
        dca_ts[t]       = dca_amt
        emerg_tgt[t]    = ef_target
        job_state_ts[t] = job_state

    return dict(
        cash=cash,
        high_debt=high_debt,
        low_debt=low_debt,
        investments=investments,
        investment_principal=investment_principal,
        income_ts=income_ts,
        expense_ts=expense_ts,
        base_expense_ts=base_expense_ts,
        net_flow_ts=net_flow_ts,
        dca_ts=dca_ts,
        emergency_target=emerg_tgt,
        job_state_ts=job_state_ts,
    )


# ══════════════════════════════════════════════════════════════════════
# DERIVED METRICS
# ══════════════════════════════════════════════════════════════════════


def compute_net_worth(res: dict) -> np.ndarray:
    """Net Worth = Cash + Investments − High Debt − Low Debt."""
    return (
        res["cash"] + res["investments"]
        - res["high_debt"] - res["low_debt"]
    )


def compute_total_assets(res: dict) -> np.ndarray:
    """Total Assets = Cash + Investments."""
    return res["cash"] + res["investments"]


def compute_target_year_expense_return_equals_investment(
    res: dict,
    annual_return: float,
    window_months: int = 12
) -> float:
    """
    计算“第2复利点”（财务自由点）的首个达成年份（中位数）。
    
    定义：
        投资账户产生的理论月利息，连续 N 个月（默认12）大于等于当月基础生活支出。
        即：Passive Income >= Base Expense (Stable for N months)

    参数：
        res (dict): 模拟结果字典，包含 'investments', 'base_expense_ts' 等。
        annual_return (float): 假设的年化收益率 (如 0.04 表示 4%)。
        window_months (int): 判定“稳定”所需的连续达标月数，默认 12 个月。

    返回：
        float: 达成目标的年份（例如 15.5 年）。若未达成返回 NaN。
    """
    # 1. 边界检查：如果收益率非正，无法产生利息，直接返回 NaN
    if annual_return <= 0:
        return np.nan

    # 2. 计算月化收益率 (几何平均)
    # 公式：(1 + r_m)^12 = 1 + r_y  =>  r_m = (1 + r_y)^(1/12) - 1
    monthly_return = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
    
    # 3. 提取数据 (Shape: [T_months + 1, N_simulations])
    # investments: 月末投资余额
    # base_expense: 基础生活支出（不含一次性大额支出）
    invest_amount = res["investments"]
    base_expense = res.get("base_expense_ts", res["expense_ts"])
    
    # 模拟时长检查
    n_months, n_sims = invest_amount.shape
    if n_months < window_months:
        return np.nan

    # 4. 计算每月的理论被动收入
    passive_income = invest_amount * monthly_return

    # 5. 生成布尔矩阵：当月是否覆盖支出
    # 只有当支出大于0时才进行判断，避免除以0或无意义的比较
    # 注意：这里要求被动收入 >= 支出
    is_covered = (base_expense > 0) & (passive_income >= base_expense)

    # 6. 使用累加和 (CumSum) 实现高效的滑动窗口检测
    # 逻辑：将布尔值转为0/1，计算累加和。
    # 窗口和 = cumsum[t] - cumsum[t - window]
    # 如果窗口和 == window，说明该窗口内所有月份均为 True（连续达标）
    
    # 转换为 int (0 或 1)
    covered_int = is_covered.astype(int)
    
    # 构造 padding，方便做切片减法
    # 在时间轴顶部加一行 0，shape 变为 [T+2, N] (因cumsum本身维度不变，需构造错位)
    cs = np.vstack([np.zeros((1, n_sims)), np.cumsum(covered_int, axis=0)])
    
    # 计算滑动窗口内的“达标月数”
    # rolling_sum[i] 代表以第 i 个月为**结束**的窗口内的达标总数
    # 数据长度将变为 n_months - window_months + 1
    rolling_sum = cs[window_months:] - cs[:-window_months]
    
    # 7. 判定是否达成稳定条件 (和等于窗口长度)
    stabilized = (rolling_sum == window_months)
    
    # 8. 寻找首次达成的时间索引
    # any_hit: 该次模拟中是否曾经达成过
    any_hit = np.any(stabilized, axis=0)
    
    # argmax: 返回第一个 True 的索引。如果全为 False，也会返回 0 (需配合 any_hit 过滤)
    first_idx = np.argmax(stabilized, axis=0)
    
    # 9. 转换为年份
    # first_idx 是窗口计算后的索引。
    # 如果 first_idx = 0，代表第 0 到 window-1 个月达标，达成时间点是第 window-1 个月（月末）
    # 对应原数组的索引需要加上 window_months - 1
    found_month_idx = first_idx + (window_months - 1)
    
    # 初始化结果数组为 NaN
    result_years = np.full(n_sims, np.nan)
    
    # 填入达成模拟的年份 (月份 / 12)
    result_years[any_hit] = found_month_idx[any_hit] / 12.0

    # 返回中位数
    return float(np.nanmedian(result_years))


def compute_target_year_dca_over_monthly_return_equals_investment(
    res: dict,
    annual_return: float,
    window_months: int = 6
) -> float:
    """
    计算“第1复利点”（收益覆盖投入点）的首个达成年份（中位数）。
    
    定义：
        投资账户产生的理论月利息，连续 N 个月（默认6）大于等于当月定投金额。
        即：Passive Income >= Monthly DCA Amount (Stable for N months)
        这意味着你的资产增值速度已经超过了你辛苦存钱的速度。

    参数：
        res (dict): 模拟结果字典，包含 'investments', 'dca_ts'。
        annual_return (float): 假设的年化收益率。
        window_months (int): 判定“稳定”所需的连续达标月数，默认 6 个月。

    返回：
        float: 达成目标的年份。若未达成返回 NaN。
    """
    # 1. 边界检查
    if annual_return <= 0:
        return np.nan

    # 2. 计算月化收益率
    monthly_return = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0

    # 3. 提取数据
    dca_ts = res["dca_ts"]         # 当月定投金额
    invest_amount = res["investments"] # 月末投资余额
    
    n_months, n_sims = invest_amount.shape
    if n_months < window_months:
        return np.nan

    # 4. 计算理论被动收入
    passive_income = invest_amount * monthly_return

    # 5. 生成布尔矩阵：利息是否覆盖定投
    # 逻辑：
    # (a) 利息 >= 定投额
    # (b) 定投额 > 0 (我们只关心依然在进行定投的时期。如果定投停止了，比较意义不大)
    #     如果希望即使停止定投（DCA=0）也算达标，可以去掉 dca_ts > 0 条件。
    #     但在财务规划中，这个指标通常用来衡量“资产滚雪球效应是否超过了人力投入”。
    is_covered = (dca_ts > 0) & (passive_income >= dca_ts)

    # 6. 滑动窗口检测 (逻辑同上)
    covered_int = is_covered.astype(int)
    cs = np.vstack([np.zeros((1, n_sims)), np.cumsum(covered_int, axis=0)])
    
    rolling_sum = cs[window_months:] - cs[:-window_months]
    
    # 7. 判定稳定达标
    stabilized = (rolling_sum == window_months)
    
    # 8. 寻找首次达成时间
    any_hit = np.any(stabilized, axis=0)
    first_idx = np.argmax(stabilized, axis=0)
    
    found_month_idx = first_idx + (window_months - 1)
    
    result_years = np.full(n_sims, np.nan)
    result_years[any_hit] = found_month_idx[any_hit] / 12.0

    return float(np.nanmedian(result_years))


def run_year10_sensitivity_analysis(
    p: dict,
    salary_min: float,
    salary_max: float,
    salary_points: int,
    return_min_pct: float,
    return_max_pct: float,
    return_points: int,
    analysis_sims: int,
) -> dict:
    """
    固定其余参数，扫描（初始工资, 年化收益率）网格，输出第10年总资产。

        输出二维矩阵 z_assets：
            行 = 初始工资，列 = 年化收益率。
    """
    wages = np.linspace(float(salary_min), float(salary_max), int(salary_points))
    returns_pct = np.linspace(
        float(return_min_pct), float(return_max_pct), int(return_points)
    )
    returns = returns_pct / 100.0

    z_assets = np.zeros((len(wages), len(returns)))
    z_target_year = np.full((len(wages), len(returns)), np.nan)
    z_first_compound_year = np.full((len(wages), len(returns)), np.nan)

    # 【核心修正】：将模拟窗口从 10 年延长到 30 年
    # 这样高收入者可能在第 5 年达成，低收入者在第 25 年达成，差距就会拉开
    sim_years = 50
    sim_months = sim_years * 12
    
    base_p = dict(p)
    base_p["n_years"] = sim_years
    base_p["n_months"] = sim_months
    base_p["n_simulations"] = int(analysis_sims)

    for i, w0 in enumerate(wages):
        for j, r in enumerate(returns):
            cur_p = dict(base_p)
            cur_p["monthly_income"] = float(w0)
            cur_p["invest_return_mean"] = float(r)

            res = run_simulation(cur_p)

            # 1. 获取第 10 年（第 120 个月）的总资产用于热力图
            # 注意：如果模拟年份不足 10 年这里会报错，但现在我们设了 30 年所以没问题
            total_assets = compute_total_assets(res)
            # 索引 120 代表第 10 年末
            if 120 < total_assets.shape[0]:
                z_assets[i, j] = np.percentile(total_assets[120, :], 50)
            else:
                z_assets[i, j] = np.percentile(total_assets[-1, :], 50)

            # 2. 计算第 2 复利点（覆盖生活支出）
            # 现在有 30 年的数据，足够区分快慢了
            z_target_year[i, j] = compute_target_year_expense_return_equals_investment(
                res, annual_return=float(r)
            )

            # 3. 计算第 1 复利点（覆盖定投）
            z_first_compound_year[i, j] = (
                compute_target_year_dca_over_monthly_return_equals_investment(
                    res, annual_return=float(r)
                )
            )

    return dict(
        wages=wages,
        returns_pct=returns_pct,
        z_assets=z_assets,
        z_target_year=z_target_year,
        z_first_compound_year=z_first_compound_year,
    )

# 尝试导入 skimage，如果失败则标记
try:
    from skimage.measure import find_contours
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def _fill_nan_2d_nearest(z: np.ndarray) -> np.ndarray:
    """对 2D 矩阵做稳健填充，避免梯度在 NaN 边界爆炸。"""
    arr = np.asarray(z, dtype=float)
    if arr.ndim != 2:
        return arr # Should handle error upstream
    
    # 如果全空，无法填充
    if np.all(np.isnan(arr)):
        return arr

    if not np.isnan(arr).any():
        return arr

    # 使用 Pandas 的插值功能来填充内部 NaN
    df = pd.DataFrame(arr)
    # 双向插值以覆盖尽可能多的区域
    df = df.interpolate(axis=0, limit_direction="both")
    df = df.interpolate(axis=1, limit_direction="both")

    filled = df.to_numpy(dtype=float)
    
    # 如果边缘仍有 NaN，用整体中位数填充（最后一道防线）
    if np.isnan(filled).any():
        med = np.nanmedian(filled)
        if np.isnan(med):
            med = 0.0
        filled = np.where(np.isnan(filled), med, filled)
    return filled


def calculate_iso_sensitivity_line(sens: dict, z_key: str) -> list:
    """
    计算等敏感度线：归一化后的 ∂n/∂w - ∂n/∂r = 0。

    返回：
        list[dict]，每条线元素为 {"x": returns_pct_array, "y": wages_array}。
    """
    if not HAS_SKIMAGE:
        # 如果没有安装 scikit-image，直接返回空，避免使用劣质的纯 NumPy 算法画出错误的线
        return []

    wages = np.asarray(sens.get("wages", []), dtype=float)
    returns_pct = np.asarray(sens.get("returns_pct", []), dtype=float)
    z_raw = np.asarray(sens.get(z_key), dtype=float)

    # 基础校验
    if wages.ndim != 1 or returns_pct.ndim != 1 or z_raw.ndim != 2:
        return []
    if z_raw.shape != (len(wages), len(returns_pct)):
        return []
    if len(wages) < 3 or len(returns_pct) < 3: # 梯度计算至少需要 2-3 个点
        return []

    # 填充 NaN
    z = _fill_nan_2d_nearest(z_raw)
    
    # 再次检查填充后是否有效
    if np.all(np.isnan(z)) or np.all(z == 0):
        return []

    # 计算范围 (Range) 用于归一化
    delta_w = float(np.nanmax(wages) - np.nanmin(wages))
    delta_r = float(np.nanmax(returns_pct) - np.nanmin(returns_pct))
    
    if delta_w <= 1e-9 or delta_r <= 1e-9:
        return []

    # 计算梯度
    # z 的行是 wages (Axis 0)，列是 returns (Axis 1)
    # np.gradient 返回 (grad_axis0, grad_axis1)
    grad_w, grad_r = np.gradient(z, wages, returns_pct, edge_order=1)

    # 计算差异矩阵 D = Grad_w_norm - Grad_r_norm
    # 归一化梯度 = 物理梯度 * 物理范围
    d = (grad_w * delta_w) - (grad_r * delta_r)

    # 处理 D 中的无限值（防止 find_contours 崩溃）
    if not np.all(np.isfinite(d)):
        finite_mask = np.isfinite(d)
        if not np.any(finite_mask):
            return []
        # 用中位数替换无穷大值
        d = np.where(finite_mask, d, np.nanmedian(d[finite_mask]))

    lines = []
    
    # 使用 skimage 寻找 0 等高线
    # find_contours 返回 list of (row, col) coordinates
    contours = find_contours(d, level=0.0)
    
    n_w, n_r = d.shape
    
    for c in contours:
        # 过滤掉太短的噪点线
        if c.shape[0] < 3:
            continue
            
        row_idx = c[:, 0] # 对应 Wages 索引
        col_idx = c[:, 1] # 对应 Returns 索引
        
        # 将索引映射回物理坐标
        # 因为我们的网格是均匀的（linspace），可以直接线性插值
        y_wage = np.interp(row_idx, np.arange(n_w), wages)
        x_ret = np.interp(col_idx, np.arange(n_r), returns_pct)
        
        lines.append({"x": x_ret, "y": y_wage})

    return lines


def _overlay_iso_sensitivity_line(fig: go.Figure, sens: dict, z_key: str, color: str = "white"):
    """在热力图上叠加等敏感度线。"""
    lines = calculate_iso_sensitivity_line(sens, z_key)
    
    if not lines:
        if not HAS_SKIMAGE:
             msg = "未安装 scikit-image，无法计算等敏感度线"
        else:
             msg = "未找到等敏感度平衡点 (单边效应主导)"
             
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            text=msg,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="gray", borderwidth=1
        )
        return fig

    for idx, line in enumerate(lines):
        fig.add_trace(
            go.Scatter(
                x=line["x"],
                y=line["y"],
                mode="lines",
                line=dict(color=color, dash="dash", width=2),
                name="等敏感度线 (Iso-sensitivity)",
                showlegend=(idx == 0), # 只显示第一个图例，避免重复
                hovertemplate="<b>等敏感度点</b><br>年化收益率: %{x:.2f}%<br>初始工资: %{y:,.0f}<extra></extra>",
            )
        )
    return fig


def plot_year10_asset_heatmap(sens: dict):
    """第10年总资产热力图（叠加基于 z_target_year 的等敏感度线）。"""
    # 注意：我们通常想看的是“达成财务自由时间”的敏感度，而不是“第10年资产”的敏感度
    # 但如果用户确实想把这条线画在资产热力图上作为参考，也是可以的。
    # 这里我们统一使用 z_target_year 来计算敏感度线，因为这更有意义。
    
    fig = go.Figure(
        go.Heatmap(
            x=sens["returns_pct"],
            y=sens["wages"],
            z=sens["z_assets"],
            colorscale="Viridis",
            colorbar_title="第10年总资产",
            hovertemplate="收益率: %{x:.1f}%<br>工资: %{y:,.0f}<br>资产: %{z:,.0f}<extra></extra>"
        )
    )
    # 叠加线
    _overlay_iso_sensitivity_line(fig, sens, z_key="z_assets", color="white")
    
    fig.update_layout(
        title="🧭 第10年总资产热力图 (虚线为复利点等敏感度线)",
        xaxis_title="年化收益率（%）",
        yaxis_title="初始工资（月）",
        template="plotly_white",
        height=520,
    )
    return fig


def plot_target_year_heatmap(sens: dict):
    """第2复利点年限热力图（含等敏感度线）。"""
    fig = go.Figure(
        go.Heatmap(
            x=sens["returns_pct"],
            y=sens["wages"],
            z=sens["z_target_year"],
            colorscale="Turbo",
            colorbar_title="达成复利点年限",
            hovertemplate="收益率: %{x:.1f}%<br>工资: %{y:,.0f}<br>年限: %{z:.1f}年<extra></extra>"
        )
    )
    # 叠加线
    _overlay_iso_sensitivity_line(fig, sens, z_key="z_target_year", color="black") # 亮色背景用黑线
    
    fig.update_layout(
        title="🗺️ 第2复利点年限热力图 (虚线为等敏感度线)",
        xaxis_title="年化收益率（%）",
        yaxis_title="初始工资（月）",
        template="plotly_white",
        height=520,
    )
    return fig

def _get_evenly_spaced_indices(total_len: int, target_count: int = 10) -> np.ndarray:
    """辅助函数：计算均匀间隔的索引，避免 np.linspace 取整导致的间隔不均问题。"""
    if total_len <= target_count:
        return np.arange(total_len)
    # 计算步长，确保间隔是整数
    step = max(1, total_len // target_count)
    return np.arange(0, total_len, step)


def plot_asset_vs_return_by_salary(sens: dict):
    """不同初始工资下：10年总资产随年化收益率变化。"""
    fig = go.Figure()

    wages = sens["wages"]
    returns_pct = sens["returns_pct"]
    z = sens["z_assets"]

    selected_idx = _get_evenly_spaced_indices(len(wages), target_count=12)
    
    for i in selected_idx:
        fig.add_trace(go.Scatter(
            x=returns_pct,
            y=z[i, :],
            mode="lines",
            name=f"工资={wages[i]:,.0f}",
        ))

    fig.update_layout(
        title="📈 不同初始工资下：第10年总资产 vs 年化收益率",
        xaxis_title="年化收益率（%）",
        yaxis_title="第10年总资产（中位数）",
        template="plotly_white",
        height=520,
    )
    return fig


def plot_asset_vs_salary_by_return(sens: dict):
    """不同年化收益率下：10年总资产随初始工资变化。"""
    fig = go.Figure()

    wages = sens["wages"]
    returns_pct = sens["returns_pct"]
    z = sens["z_assets"]

    selected_idx = _get_evenly_spaced_indices(len(returns_pct), target_count=12)

    for j in selected_idx:
        fig.add_trace(go.Scatter(
            x=wages,
            y=z[:, j],
            mode="lines",
            name=f"收益率={returns_pct[j]:.1f}%",
        ))

    fig.update_layout(
        title="📈 不同年化收益率下：第10年总资产 vs 初始工资",
        xaxis_title="初始工资（月）",
        yaxis_title="第10年总资产（中位数）",
        template="plotly_white",
        height=520,
    )
    return fig


def plot_target_year_vs_return_by_salary(sens: dict):
    """不同初始工资下：第2复利点（稳定覆盖）年限随年化收益率变化。"""
    fig = go.Figure()

    wages = sens["wages"]
    returns_pct = sens["returns_pct"]
    z = sens["z_target_year"]
    mask_from_3pct = returns_pct >= 3.0
    x_plot = returns_pct[mask_from_3pct]

    selected_idx = _get_evenly_spaced_indices(len(wages), target_count=12)

    for i in selected_idx:
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=z[i, mask_from_3pct],
            mode="lines+markers",
            name=f"工资={wages[i]:,.0f}",
        ))

    fig.update_layout(
        title="⏳ 不同初始工资下：第2复利点年限（稳定覆盖基础生活支出）vs 年化收益率",
        xaxis_title="年化收益率（%）",
        yaxis_title="第2复利点年限（年，稳定覆盖口径）",
        template="plotly_white",
        height=520,
    )
    return fig


def plot_target_year_vs_salary_by_return(sens: dict):
    """不同年化收益率下：第2复利点（稳定覆盖）年限随初始工资变化。"""
    fig = go.Figure()

    wages = sens["wages"]
    returns_pct = sens["returns_pct"]
    z = sens["z_target_year"]

    selected_idx = _get_evenly_spaced_indices(len(returns_pct), target_count=12)

    for j in selected_idx:
        fig.add_trace(go.Scatter(
            x=wages,
            y=z[:, j],
            mode="lines+markers",
            name=f"收益率={returns_pct[j]:.1f}%",
        ))

    fig.update_layout(
        title="⏳ 不同年化收益率下：第2复利点年限（稳定覆盖基础生活支出）vs 初始工资",
        xaxis_title="初始工资（月）",
        yaxis_title="第2复利点年限（年，稳定覆盖口径）",
        template="plotly_white",
        height=520,
    )
    return fig


def plot_first_compound_year_vs_return_by_salary(sens: dict):
    """不同初始工资下：第1复利点年限随年化收益率变化。"""
    fig = go.Figure()

    wages = sens["wages"]
    returns_pct = sens["returns_pct"]
    z = sens["z_first_compound_year"]

    selected_idx = _get_evenly_spaced_indices(len(wages), target_count=12)

    for i in selected_idx:
        fig.add_trace(go.Scatter(
            x=returns_pct,
            y=z[i, :],
            mode="lines+markers",
            name=f"工资={wages[i]:,.0f}",
        ))

    fig.update_layout(
        title="🔹 不同初始工资下：第1复利点年限 vs 年化收益率",
        xaxis_title="年化收益率（%）",
        yaxis_title="第1复利点年限（年）",
        template="plotly_white",
        height=520,
    )
    return fig


def plot_first_compound_year_vs_salary_by_return(sens: dict):
    """不同年化收益率下：第1复利点年限随初始工资变化。"""
    fig = go.Figure()

    wages = sens["wages"]
    returns_pct = sens["returns_pct"]
    z = sens["z_first_compound_year"]

    selected_idx = _get_evenly_spaced_indices(len(returns_pct), target_count=12)

    for j in selected_idx:
        fig.add_trace(go.Scatter(
            x=wages,
            y=z[:, j],
            mode="lines+markers",
            name=f"收益率={returns_pct[j]:.1f}%",
        ))

    fig.update_layout(
        title="🔹 不同年化收益率下：第1复利点年限 vs 初始工资",
        xaxis_title="初始工资（月）",
        yaxis_title="第1复利点年限（年）",
        template="plotly_white",
        height=520,
    )
    return fig


def compute_metrics(res: dict, p: dict) -> dict:
    """Scalar summary statistics across all N simulations."""
    N  = p["n_simulations"]
    nw = compute_net_worth(res)

    # Bankruptcy: high debt ever exceeds 2× starting (or 500 k floor)
    bankr_thresh = max(p["high_debt_init"] * 2, 500_000)
    prob_bankruptcy = (
        np.any(res["high_debt"] > bankr_thresh, axis=0).mean() * 100.0
    )

    # Median month to high-debt-free
    hd = res["high_debt"]
    debt_free_m = np.full(N, np.nan)
    for s in range(N):
        hits = np.where(hd[:, s] < 1.0)[0]
        if hits.size:
            debt_free_m[s] = hits[0]
    median_debt_free = np.nanmedian(debt_free_m)

    # Median month to ALL debt free (high + low)
    total_d = res["high_debt"] + res["low_debt"]
    all_free_m = np.full(N, np.nan)
    for s in range(N):
        hits = np.where(total_d[:, s] < 1.0)[0]
        if hits.size:
            all_free_m[s] = hits[0]
    median_all_free = np.nanmedian(all_free_m)

    # Median month to positive net worth
    pos_nw_m = np.full(N, np.nan)
    for s in range(N):
        hits = np.where(nw[:, s] > 0)[0]
        if hits.size:
            pos_nw_m[s] = hits[0]
    median_pos_nw = np.nanmedian(pos_nw_m)

    never_clear = np.isnan(debt_free_m).mean() * 100.0
    final_nw    = nw[-1, :]

    # ── v3 new metrics ───────────────────────────────────────────────
    js = res["job_state_ts"]

    # % of sims that ever enter survival mode
    pct_ever_survival = (
        np.any(js == JOB_SURVIVAL, axis=0).mean() * 100.0
    )

    # Average total months spent unemployed per sim
    avg_months_unemployed = np.mean(
        np.sum(js == JOB_UNEMPLOYED, axis=0),
    )

    # Average total months spent in survival per sim
    avg_months_survival = np.mean(
        np.sum(js == JOB_SURVIVAL, axis=0),
    )

    return dict(
        prob_bankruptcy=prob_bankruptcy,
        median_debt_free_months=median_debt_free,
        median_all_debt_free_months=median_all_free,
        median_pos_nw_months=median_pos_nw,
        never_clear_high_pct=never_clear,
        final_nw_p10=np.percentile(final_nw, 10),
        final_nw_p50=np.percentile(final_nw, 50),
        final_nw_p90=np.percentile(final_nw, 90),
        # v3
        pct_ever_survival=pct_ever_survival,
        avg_months_unemployed=avg_months_unemployed,
        avg_months_survival=avg_months_survival,
    )


# ══════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════

COLORS = dict(
    p10="#636EFA", p50="#EF553B", p90="#00CC96",
    fill="rgba(99,110,250,0.15)",
    debt="#FF6692", debt_fill="rgba(255,102,146,0.15)",
    income="#AB63FA", expense="#FFA15A",
)


def _x_years(T):
    return np.arange(T + 1) / 12.0


def _pct_traces(data, x, name, col, cfill):
    p10 = np.percentile(data, 10, axis=1)
    p50 = np.percentile(data, 50, axis=1)
    p90 = np.percentile(data, 90, axis=1)
    return [
        go.Scatter(
            x=x, y=p90, mode="lines", name=f"{name} P90",
            line=dict(color=col, width=1, dash="dot"),
        ),
        go.Scatter(
            x=x, y=p10, mode="lines", name=f"{name} P10",
            line=dict(color=col, width=1, dash="dot"),
            fill="tonexty", fillcolor=cfill,
        ),
        go.Scatter(
            x=x, y=p50, mode="lines", name=f"{name} Median",
            line=dict(color=col, width=2.5),
        ),
    ]


def _layout(title, xt="距今年份", yt="金额", h=480):
    return dict(
        title=title, xaxis_title=xt, yaxis_title=yt,
        template="plotly_white", height=h,
        legend=dict(orientation="h", y=-0.18),
        yaxis=dict(tickformat=","),
    )


# ── Individual chart functions ───────────────────────────────────────

def plot_net_worth(res, p):
    nw = compute_net_worth(res)
    x  = _x_years(p["n_months"])
    fig = go.Figure()
    for tr in _pct_traces(nw, x, "净资产", COLORS["p50"], COLORS["fill"]):
        fig.add_trace(tr)
    fig.add_hline(
        y=0, line_dash="dash", line_color="grey", opacity=0.5,
        annotation_text="盈亏平衡",
    )
    fig.update_layout(**_layout("📈 净资产（P10 / 中位数 / P90）", xt="距今年份", yt="金额"))
    return fig


def plot_high_debt(res, p):
    x = _x_years(p["n_months"])
    fig = go.Figure()
    for tr in _pct_traces(
        res["high_debt"], x, "高息负债",
        COLORS["debt"], COLORS["debt_fill"],
    ):
        fig.add_trace(tr)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.update_layout(**_layout("💳 高息负债", xt="距今年份", yt="金额"))
    return fig


def plot_low_debt(res, p):
    x = _x_years(p["n_months"])
    fig = go.Figure()
    for tr in _pct_traces(
        res["low_debt"], x, "软负债",
        "#FFA15A", "rgba(255,161,90,0.15)",
    ):
        fig.add_trace(tr)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.update_layout(**_layout("🤝 软负债 / 低息负债", xt="距今年份", yt="金额"))
    return fig


def plot_cash_flow(res, p):
    x   = _x_years(p["n_months"])
    inc = np.percentile(res["income_ts"],   50, axis=1)
    exp = np.percentile(res["expense_ts"],  50, axis=1)
    nf  = np.percentile(res["net_flow_ts"], 50, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=inc, mode="lines", name="收入（中位数）",
        line=dict(color=COLORS["income"], width=2),
        fill="tozeroy", fillcolor="rgba(171,99,250,0.10)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=exp, mode="lines", name="支出（中位数）",
        line=dict(color=COLORS["expense"], width=2),
        fill="tozeroy", fillcolor="rgba(255,161,90,0.10)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=nf, mode="lines", name="净现金流（中位数）",
        line=dict(color="#19D3F3", width=2, dash="dash"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.4)
    fig.update_layout(**_layout(
        "💵 月度现金流（中位数）", xt="距今年份", yt="月金额", h=440,
    ))
    return fig


def plot_emergency_fund(res, p):
    x      = _x_years(p["n_months"])
    cash_m = np.percentile(res["cash"], 50, axis=1)
    tgt_m  = np.percentile(res["emergency_target"], 50, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=cash_m, mode="lines", name="现金（中位数）",
        line=dict(color="#636EFA", width=2),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.10)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=tgt_m, mode="lines", name="应急金目标",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))
    fig.update_layout(**_layout("🛡️ 应急金与目标对比", xt="距今年份", yt="金额", h=420))
    return fig


def plot_components(res, p):
    x = _x_years(p["n_months"])
    invest_p50 = np.percentile(res["investments"], 50, axis=1)
    principal_p50 = np.percentile(res["investment_principal"], 50, axis=1)
    interest_p50 = invest_p50 - principal_p50
    interest_pos = np.maximum(interest_p50, 0.0)
    interest_neg = np.maximum(-interest_p50, 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=principal_p50,
        mode="lines", name="投资本金",
        stackgroup="pos", line=dict(width=0.5, color="#00CC96"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=interest_pos,
        mode="lines", name="投资利息",
        stackgroup="pos", line=dict(width=0.5, color="#19D3F3"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=np.percentile(res["cash"], 50, axis=1),
        mode="lines", name="现金",
        stackgroup="pos", line=dict(width=0.5, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=-interest_neg,
        mode="lines", name="投资利息（负）",
        stackgroup="neg", line=dict(width=0.5, color="#2A3F5F"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=-np.percentile(res["high_debt"], 50, axis=1),
        mode="lines", name="高息负债（负）",
        stackgroup="neg", line=dict(width=0.5, color="#EF553B"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=-np.percentile(res["low_debt"], 50, axis=1),
        mode="lines", name="软负债（负）",
        stackgroup="neg", line=dict(width=0.5, color="#FFA15A"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.update_layout(**_layout("🧩 资产构成（中位数）", xt="距今年份", yt="金额", h=440))
    return fig


def plot_job_states(res, p):
    """Stacked area chart: % of simulations in each job state over time."""
    x  = _x_years(p["n_months"])
    js = res["job_state_ts"]

    stay_pct  = np.mean(js == JOB_STAY,       axis=1) * 100
    hop_pct   = np.mean(js == JOB_HOP,        axis=1) * 100
    unemp_pct = np.mean(js == JOB_UNEMPLOYED, axis=1) * 100
    surv_pct  = np.mean(js == JOB_SURVIVAL,   axis=1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=stay_pct, mode="lines", name="苟着（Stay）",
        stackgroup="one",
        line=dict(width=0.5, color="#00CC96"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=hop_pct, mode="lines", name="跳槽（Job Hop）",
        stackgroup="one",
        line=dict(width=0.5, color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=unemp_pct, mode="lines", name="失业（Unemployed）",
        stackgroup="one",
        line=dict(width=0.5, color="#EF553B"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=surv_pct, mode="lines", name="兜底（Survival）",
        stackgroup="one",
        line=dict(width=0.5, color="#FFA15A"),
    ))
    fig.update_layout(**_layout(
        "👔 职业状态分布（模拟占比）",
        xt="距今年份", yt="模拟占比（%）", h=420,
    ))
    return fig


def plot_debt_heatmap(res, p):
    hd   = res["high_debt"]
    T, N = p["n_months"], p["n_simulations"]
    yidx = list(range(0, T + 1, 12))
    ylbl = [f"Yr {i // 12}" for i in yidx]
    mx   = max(np.percentile(hd, 99), 1)
    bins = np.linspace(0, mx, 25)
    xlbl = [f"{int(bins[i] / 1000)}k" for i in range(len(bins) - 1)]
    heat = np.zeros((len(yidx), len(bins) - 1))
    for ri, mi in enumerate(yidx):
        c, _ = np.histogram(hd[mi, :], bins=bins)
        heat[ri] = c / N * 100.0
    fig = go.Figure(go.Heatmap(
        z=heat, x=xlbl, y=ylbl,
        colorscale="YlOrRd", colorbar_title="模拟占比（%）",
    ))
    fig.update_layout(
        title="🔥 高息负债分布热力图",
        xaxis_title="负债水平", yaxis_title="时间",
        template="plotly_white", height=460,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════


def main():
    st.title("🎲 蒙特卡洛财务规划器 v3")
    st.caption(
        "状态依赖模拟：包含 **职业状态机**（苟着/跳槽/失业/兜底）、"
        "**动态支出弹性**、**赤字融资策略** 与 **参数持久化**。"
    )

    params = build_sidebar_inputs()

    run = st.sidebar.button(
        "🚀 运行模拟", type="primary", use_container_width=True,
    )

    if run:
        st.session_state["params"] = params
        with st.spinner(
            f"正在运行 {params['n_simulations']:,} 次模拟 × "
            f"{params['n_years']} 年 …"
        ):
            st.session_state["results"] = run_simulation(params)
            st.session_state["metrics"] = compute_metrics(
                st.session_state["results"], params,
            )

    if "results" not in st.session_state:
        st.info("👈 请先配置参数，然后点击 **运行模拟**。")
        return

    res = st.session_state["results"]
    met = st.session_state["metrics"]
    par = st.session_state["params"]

    # ── Key Metrics ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 关键指标")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("破产风险", f"{met['prob_bankruptcy']:.1f}%")
    with c2:
        v = met["median_debt_free_months"]
        st.metric(
            "高息债清零",
            f"{v / 12:.1f} 年" if not np.isnan(v) else "未达成",
        )
    with c3:
        v = met["median_all_debt_free_months"]
        st.metric(
            "全部债务清零",
            f"{v / 12:.1f} 年" if not np.isnan(v) else "未达成",
        )
    with c4:
        v = met["median_pos_nw_months"]
        st.metric(
            "净资产转正",
            f"{v / 12:.1f} 年" if not np.isnan(v) else "未达成",
        )
    with c5:
        st.metric(
            "高息债始终未清",
            f"{met['never_clear_high_pct']:.1f}%",
        )

    # ── v3 Job State Metrics ─────────────────────────────────────────
    c6, c7, c8 = st.columns(3)
    with c6:
        st.metric(
            "曾进入兜底模式",
            f"{met['pct_ever_survival']:.1f}%",
        )
    with c7:
        st.metric(
            "平均失业月数",
            f"{met['avg_months_unemployed']:.1f}",
        )
    with c8:
        st.metric(
            "平均兜底月数",
            f"{met['avg_months_survival']:.1f}",
        )

    st.markdown("---")
    st.subheader(f"💰 第 {par['n_years']} 年末净资产")
    c1, c2, c3 = st.columns(3)
    c1.metric("悲观（P10）", f"{met['final_nw_p10']:,.0f}")
    c2.metric("中位（P50）", f"{met['final_nw_p50']:,.0f}")
    c3.metric("乐观（P90）", f"{met['final_nw_p90']:,.0f}")

    # ── Charts ───────────────────────────────────────────────────────
    st.markdown("---")
    st.plotly_chart(plot_net_worth(res, par), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_high_debt(res, par), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_low_debt(res, par), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(plot_cash_flow(res, par), use_container_width=True)
    with col_d:
        st.plotly_chart(
            plot_emergency_fund(res, par), use_container_width=True,
        )

    st.plotly_chart(plot_components(res, par), use_container_width=True)

    # ── v3 new chart ─────────────────────────────────────────────────
    st.plotly_chart(plot_job_states(res, par), use_container_width=True)

    st.plotly_chart(plot_debt_heatmap(res, par), use_container_width=True)

    # ── 第10年总资产敏感性分析 ───────────────────────────────────────
    st.markdown("---")
    st.subheader("🧭 第10年总资产敏感性分析")
    st.caption(
        "固定其他参数，扫描不同的初始工资与年化收益率："
        "绘制八张关系图："
        "① 第10年总资产热力图（叠加等敏感度线）；"
        "② 第2复利点年限热力图（叠加等敏感度线）；"
        "③ 不同工资下总资产随年化收益率变化；"
        "④ 不同年化收益率下总资产随工资变化；"
        "⑤ 不同工资下第2复利点年限（稳定覆盖基础生活支出）随收益率变化；"
        "⑥ 不同收益率下第2复利点年限（稳定覆盖基础生活支出）随工资变化；"
        "⑦ 不同工资下第1复利点年限随收益率变化；"
        "⑧ 不同收益率下第1复利点年限随工资变化。"
    )

    csa1, csa2, csa3 = st.columns(3)
    with csa1:
        sal_min = st.number_input("初始工资最小值", 0, 1_000_000, 20_000, step=1_000)
        sal_max = st.number_input("初始工资最大值", 1_000, 2_000_000, 50_000, step=1_000)
        sal_pts = st.slider("工资网格点数", 8, 40, 16)
    with csa2:
        r_min = st.slider("年化收益率最小值（%）", -10.0, 30.0, 0.0, 0.5)
        r_max = st.slider("年化收益率最大值（%）", -5.0, 60.0, 20.0, 0.5)
        r_pts = st.slider("收益率网格点数", 8, 40, 11)
    with csa3:
        sims_for_sens = st.slider(
            "敏感性分析模拟次数", 100, 10000,
            min(400, par["n_simulations"]), step=500,
            help="该分析会做网格扫描；建议用较小模拟次数以提高速度。",
        )

    run_sens = st.button("📈 运行第10年敏感性分析", use_container_width=True)
    if run_sens:
        if sal_max <= sal_min:
            st.warning("初始工资最大值必须大于最小值。")
        elif r_max <= r_min:
            st.warning("年化收益率最大值必须大于最小值。")
        else:
            with st.spinner("正在扫描网格并计算资产/年限曲线，请稍候…"):
                st.session_state["sens_year10"] = run_year10_sensitivity_analysis(
                    par,
                    salary_min=sal_min,
                    salary_max=sal_max,
                    salary_points=sal_pts,
                    return_min_pct=r_min,
                    return_max_pct=r_max,
                    return_points=r_pts,
                    analysis_sims=sims_for_sens,
                )

    if "sens_year10" in st.session_state:
        sens = st.session_state["sens_year10"]
        st.plotly_chart(plot_year10_asset_heatmap(sens), use_container_width=True)
        st.plotly_chart(plot_target_year_heatmap(sens), use_container_width=True)
        st.plotly_chart(plot_asset_vs_return_by_salary(sens), use_container_width=True)
        st.plotly_chart(plot_asset_vs_salary_by_return(sens), use_container_width=True)
        st.plotly_chart(plot_target_year_vs_return_by_salary(sens), use_container_width=True)
        st.plotly_chart(plot_target_year_vs_salary_by_return(sens), use_container_width=True)
        st.plotly_chart(plot_first_compound_year_vs_return_by_salary(sens), use_container_width=True)
        st.plotly_chart(plot_first_compound_year_vs_salary_by_return(sens), use_container_width=True)

    # ── Waterfall Explanation (updated for v3) ───────────────────────
    st.markdown("---")
    st.subheader("🌊 现金流瀑布规则（v3）")
    st.markdown("""
    每个月的**正向结余**按如下优先级分配：

    | # | 去向 | 规则 |
    |:-:|------|------|
    | 1 | **生活支出** | 基础支出 × 通胀 + 生活膨胀 ± 生存模式系数 |
    | 2 | **高息负债** | 偿还本金（利息按月末计提） |
    | 3 | **软债摊还** | 固定月度摊还，直到清零 |
    | 4 | **应急金** | 现金补足到 *N* × 月支出 |
    | 5 | **投资（定投）** | 从 *定投起始年* 开始，最多投入 *(收入−开销) × 定投比例* |

    **赤字融资**（当支出 > 收入）：

    | 层级 | 资金来源 | 说明 |
    |:----:|----------|------|
    | 1 | **应急金（现金）** | 第一缓冲 |
    | 2 | **卖出投资** | 按市值变现 |
    | 3 | **新增软债** | 亲友借款，最多到 *软债上限* |
    | 4 | **新增高息债** | 信用卡/消费贷 —— *债务螺旋* |
    """)

    st.markdown("---")
    st.subheader("🔄 职业状态机")
    st.markdown("""
    | 状态 | 描述 | 收入 |
    |:----:|------|------|
    | **A – 苟着（Stay）** | 停滞，缓慢增长≈通胀 | 全额工资 |
    | **B – 跳槽（Hop）** | 岗位切换，工资瞬时跃升 | 跃升后工资 |
    | **C – 失业（Unemployed）** | 无固定工作，低保障收入 | *unemp_income_pct* × 原收入 |
    | **D – 兜底（Survival）** | 打零工/外卖等 | 兜底工资下限 |

    **状态转移**（月度马尔可夫链）：
    - **苟着 → 跳槽**：工资跃升，次月回到苟着
    - **苟着 → 失业**：发生裁员/离职
    - **失业 → 苟着**：再就业（带折损）
    - **失业 → 兜底**：长期失业退化到生存模式
    - **兜底 → 苟着**：恢复正常就业
    """)

    # ── Download ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 下载汇总数据")
    nw = compute_net_worth(res)
    df = pd.DataFrame({
        "月份":        np.arange(par["n_months"] + 1),
        "年份":        np.arange(par["n_months"] + 1) / 12.0,
        "净资产_P10": np.percentile(nw, 10, axis=1),
        "净资产_P50": np.percentile(nw, 50, axis=1),
        "净资产_P90": np.percentile(nw, 90, axis=1),
        "高息负债_P50": np.percentile(res["high_debt"], 50, axis=1),
        "软负债_P50":  np.percentile(res["low_debt"],  50, axis=1),
        "现金_P50":     np.percentile(res["cash"],       50, axis=1),
        "投资总额_P50":   np.percentile(res["investments"], 50, axis=1),
        "投资本金_P50": np.percentile(res["investment_principal"], 50, axis=1),
        "投资利息_P50": (
            np.percentile(res["investments"], 50, axis=1)
            - np.percentile(res["investment_principal"], 50, axis=1)
        ),
        "收入_P50":   np.percentile(res["income_ts"],  50, axis=1),
        "支出_P50":  np.percentile(res["expense_ts"], 50, axis=1),
        "苟着占比(%)":      np.mean(res["job_state_ts"] == JOB_STAY, axis=1) * 100,
        "失业占比(%)":     np.mean(res["job_state_ts"] == JOB_UNEMPLOYED, axis=1) * 100,
        "兜底占比(%)":  np.mean(res["job_state_ts"] == JOB_SURVIVAL, axis=1) * 100,
    })
    st.dataframe(df.head(60), use_container_width=True)
    st.download_button(
        "⬇️ 下载完整 CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="monte_carlo_v3.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
