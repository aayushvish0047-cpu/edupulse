"""
EduPulse - FastAPI service.

Endpoints
    GET  /api/health    health check
    GET  /api/careers   career roles and their subject benchmarks (for UI dropdowns)
    POST /api/analyze   full pipeline: prediction, risk tier, academic gap,
                        skill gaps, 4-week recovery plan, score explanations
    POST /api/simulate  What-If: re-predict with new attendance / study hours

Run (from the backend/ folder):
    uvicorn main:app --reload --port 8000

The model is trained on SYNTHETIC data (see train.py). Predictions are
illustrative, not validated on real students.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("edupulse")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

# Feature order is a contract with train.py.
FEATURES = [
    "grade_10",
    "grade_12",
    "current_cgpa",
    "attendance",
    "midterm_score",
    "study_hours_weekly",
]

# label, unit suffix (used in explanations)
FEATURE_META = {
    "grade_10": ("Grade 10 marks", "%"),
    "grade_12": ("Grade 12 marks", "%"),
    "current_cgpa": ("Current CGPA", ""),
    "attendance": ("Attendance", "%"),
    "midterm_score": ("Midterm score", " marks"),
    "study_hours_weekly": ("Weekly study hours", " hrs"),
}

# Ranges the model saw during training; inputs outside them get a warning.
TRAINING_RANGES = {
    "grade_10": (50, 100),
    "grade_12": (50, 100),
    "current_cgpa": (4.0, 10.0),
    "attendance": (45, 100),
    "midterm_score": (25, 100),
    "study_hours_weekly": (1, 35),
}

CGPA_MIN, CGPA_MAX = 3.0, 10.0

# Risk tiers
LOW_RISK_MIN = 7.5
MODERATE_MIN = 6.0

# Academic-gap assumptions (documented in AcademicGap.message and the response):
#   * finals contribute FINALS_WEIGHT of the final CGPA
#   * exam marks convert to grade points on a 10-point scale (marks / 10)
#   * the model's prediction implicitly assumes finals ~ the midterm score
FINALS_WEIGHT = 0.40
POINTS_PER_MARK = 0.10

MIN_DRIVER_IMPACT = 0.01  # ignore impacts smaller than this (noise)

# ---------------------------------------------------------------------------
# Career competency matrix: role -> {subject: minimum benchmark mark}
# ---------------------------------------------------------------------------
CAREER_MATRIX: dict[str, dict[str, int]] = {
    "Software Engineer": {
        "Data Structures": 70,
        "Algorithms": 70,
        "DBMS": 65,
        "Operating Systems": 60,
    },
    "Data Analyst": {
        "Statistics": 70,
        "DBMS": 70,
        "Mathematics": 65,
        "Python Programming": 65,
    },
    "Web Developer": {
        "Web Technologies": 70,
        "DBMS": 60,
        "Data Structures": 60,
        "Computer Networks": 55,
    },
    "AI/ML Engineer": {
        "Mathematics": 75,
        "Statistics": 70,
        "Python Programming": 70,
        "Machine Learning": 70,
        "Data Structures": 70,
    },
}

ROLE_LOOKUP = {role.lower(): role for role in CAREER_MATRIX}

# Alias -> canonical lowercase subject name (so "Maths", "OS" etc. still match)
SUBJECT_ALIASES = {
    "data structure": "data structures",
    "algorithm": "algorithms",
    "database": "dbms",
    "databases": "dbms",
    "database management systems": "dbms",
    "database management system": "dbms",
    "os": "operating systems",
    "operating system": "operating systems",
    "math": "mathematics",
    "maths": "mathematics",
    "engineering mathematics": "mathematics",
    "stats": "statistics",
    "python": "python programming",
    "web technology": "web technologies",
    "web tech": "web technologies",
    "cn": "computer networks",
    "networks": "computer networks",
    "networking": "computer networks",
    "ml": "machine learning",
}

PRACTICE_HINTS = {
    "Data Structures": "implement a stack, queue and hash map from scratch and solve 5 interview-style problems",
    "Algorithms": "solve a mixed set of sorting, graph and dynamic-programming problems under time limits",
    "DBMS": "design a normalised schema and write joins, aggregations and sub-queries against it",
    "Operating Systems": "work through process-scheduling and memory-management numericals",
    "Mathematics": "solve applied linear algebra, calculus and probability problem sets",
    "Statistics": "analyse a small dataset: summary statistics, distributions and one hypothesis test",
    "Python Programming": "write scripts using functions, classes, pandas and NumPy on a real dataset",
    "Web Technologies": "build a small responsive multi-page site with JavaScript interactivity",
    "Computer Networks": "trace a web request through the TCP/IP layers and solve subnetting problems",
    "Machine Learning": "train and evaluate a simple model on a public dataset and write up the results",
}

ROLE_PROJECTS = {
    "Software Engineer": "a REST API with a database, tests and documentation",
    "Data Analyst": "an end-to-end analysis and dashboard on a public dataset",
    "Web Developer": "a deployed full-stack web app with authentication",
    "AI/ML Engineer": "an end-to-end ML project with proper evaluation and a short write-up",
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
Marks = Annotated[float, Field(ge=0, le=100)]
CGPAValue = Annotated[float, Field(ge=0, le=10)]
Hours = Annotated[float, Field(ge=0, le=168)]


class StudentRequest(BaseModel):
    grade_10: Marks
    grade_12: Marks
    current_cgpa: CGPAValue
    attendance: Marks
    midterm_score: Marks
    study_hours_weekly: Hours
    target_cgpa: CGPAValue
    career_goal: str
    subject_scores: dict[str, Marks] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "grade_10": 78,
                "grade_12": 74,
                "current_cgpa": 6.8,
                "attendance": 72,
                "midterm_score": 58,
                "study_hours_weekly": 10,
                "target_cgpa": 8.0,
                "career_goal": "Software Engineer",
                "subject_scores": {
                    "Data Structures": 62,
                    "Algorithms": 68,
                    "DBMS": 70,
                },
            }
        }
    }

    @field_validator("career_goal")
    @classmethod
    def validate_career_goal(cls, value: str) -> str:
        role = ROLE_LOOKUP.get(value.strip().lower())
        if role is None:
            raise ValueError(
                f"Unknown career_goal '{value}'. Choose one of: {', '.join(CAREER_MATRIX)}"
            )
        return role


class Driver(BaseModel):
    feature: str
    label: str
    student_value: float
    cohort_mean: float
    impact_cgpa: float  # signed change in predicted CGPA vs. an average-value student
    explanation: str


class Explanation(BaseModel):
    positive_drivers: list[Driver]
    risk_drivers: list[Driver]


class AcademicGap(BaseModel):
    target_cgpa: float
    predicted_cgpa: float
    cgpa_gap: float  # target - predicted (positive = shortfall)
    on_track: bool
    required_final_score: Optional[float] = None  # out of 100
    achievable: Optional[bool] = None  # False if required score exceeds 100
    baseline_exam_score: float
    finals_weight: float
    message: str


class SkillGap(BaseModel):
    subject: str
    required_score: float
    your_score: Optional[float] = None
    shortfall: Optional[float] = None
    status: str  # "below_benchmark" | "not_assessed"
    severity: str  # "critical" | "high" | "moderate" | "unassessed"


class Strength(BaseModel):
    subject: str
    required_score: float
    your_score: float
    margin: float


class WeekPlan(BaseModel):
    week: int
    theme: str
    focus_subjects: list[str]
    actions: list[str]
    milestone: str
    suggested_hours: int


class AnalysisResponse(BaseModel):
    predicted_cgpa: float
    risk_tier: str
    academic_gap: AcademicGap
    career_goal: str
    career_readiness_pct: int
    skill_gaps: list[SkillGap]
    strengths: list[Strength]
    recovery_plan: list[WeekPlan]
    explanation: Explanation
    warnings: list[str]


class SimulationRequest(BaseModel):
    # Fixed profile
    grade_10: Marks
    grade_12: Marks
    current_cgpa: CGPAValue
    midterm_score: Marks
    # Student's actual habits (baseline)
    attendance: Marks
    study_hours_weekly: Hours
    # Slider values; omitted values fall back to the baseline
    new_attendance: Optional[Marks] = None
    new_study_hours_weekly: Optional[Hours] = None
    target_cgpa: Optional[CGPAValue] = None


class SimulationScenario(BaseModel):
    attendance: float
    study_hours_weekly: float


class SimulationResponse(BaseModel):
    baseline_cgpa: float
    simulated_cgpa: float
    cgpa_change: float
    baseline_risk_tier: str
    risk_tier: str
    scenario: SimulationScenario
    academic_gap: Optional[AcademicGap] = None
    warnings: list[str]


# ---------------------------------------------------------------------------
# App + model lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    pipeline = joblib.load(MODEL_PATH)

    trained_features = list(getattr(pipeline, "feature_names_in_", []))
    if trained_features != FEATURES:
        raise RuntimeError(
            f"Model features {trained_features} do not match API features {FEATURES}. Retrain the model."
        )

    # Single-row inference is faster without a thread pool.
    pipeline.named_steps["model"].n_jobs = 1

    app.state.pipeline = pipeline
    app.state.feature_means = dict(zip(FEATURES, pipeline.named_steps["scaler"].mean_))
    logger.info("Model loaded from %s", MODEL_PATH)
    yield


app = FastAPI(
    title="EduPulse API",
    description="Explainable academic trajectory and career skill-gap analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline(request: Request):
    return request.app.state.pipeline


def get_feature_means(request: Request) -> dict[str, float]:
    return request.app.state.feature_means


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------
def _predict_raw(pipeline, rows: list[dict]) -> np.ndarray:
    frame = pd.DataFrame(rows, columns=FEATURES)
    return pipeline.predict(frame)


def predict_cgpa(pipeline, features: dict) -> float:
    """Model inference: predicted final CGPA, clipped to the valid scale."""
    raw = float(_predict_raw(pipeline, [features])[0])
    return round(float(np.clip(raw, CGPA_MIN, CGPA_MAX)), 2)


def assign_risk_tier(predicted_cgpa: float) -> str:
    if predicted_cgpa >= LOW_RISK_MIN:
        return "Low Risk / Good Standing"
    if predicted_cgpa >= MODERATE_MIN:
        return "Moderate Concern"
    return "High Academic Risk"


def compute_academic_gap(target_cgpa: float, predicted_cgpa: float, midterm_score: float) -> AcademicGap:
    """
    cgpa_gap = target - predicted. If positive, estimate the finals score needed.

    Assumptions: finals are FINALS_WEIGHT of the final CGPA, marks convert to
    grade points as marks/10, and the model's prediction already assumes a
    finals score equal to the midterm score. Closing a gap g therefore needs
    finals = midterm + g / (FINALS_WEIGHT * POINTS_PER_MARK) marks.
    """
    gap = round(target_cgpa - predicted_cgpa, 2)
    common = dict(
        target_cgpa=round(target_cgpa, 2),
        predicted_cgpa=predicted_cgpa,
        cgpa_gap=gap,
        baseline_exam_score=round(midterm_score, 1),
        finals_weight=FINALS_WEIGHT,
    )

    if gap <= 0:
        return AcademicGap(
            **common,
            on_track=True,
            message=(
                f"Projected CGPA {predicted_cgpa:.2f} meets your {target_cgpa:.2f} target "
                f"with a {abs(gap):.2f} margin. Keep your current habits."
            ),
        )

    required = midterm_score + gap / (FINALS_WEIGHT * POINTS_PER_MARK)
    achievable = required <= 100
    if achievable:
        message = (
            f"To bridge the {gap:.2f}-point gap you need about {required:.1f}/100 in your remaining "
            f"finals (your midterm baseline is {midterm_score:.0f})."
        )
    else:
        message = (
            f"Bridging the {gap:.2f}-point gap would need about {required:.1f}/100 in finals, which is "
            "above the maximum. Finals alone cannot close it: improve attendance and study habits "
            "(try the simulator) or revisit the target."
        )
    return AcademicGap(
        **common,
        on_track=False,
        required_final_score=round(required, 1),
        achievable=achievable,
        message=message,
    )


def _norm_subject(name: str) -> str:
    key = " ".join(name.lower().split())
    return SUBJECT_ALIASES.get(key, key)


def _severity(shortfall: float) -> str:
    if shortfall >= 20:
        return "critical"
    if shortfall >= 10:
        return "high"
    return "moderate"


def analyze_skill_gaps(career_goal: str, subject_scores: dict[str, float]):
    """Compare subject scores to the role benchmarks."""
    benchmarks = CAREER_MATRIX[career_goal]
    provided = {_norm_subject(name): float(score) for name, score in subject_scores.items()}
    benchmark_keys = {_norm_subject(subject) for subject in benchmarks}

    gaps: list[SkillGap] = []
    strengths: list[Strength] = []
    for subject, required in benchmarks.items():
        score = provided.get(_norm_subject(subject))
        if score is None:
            gaps.append(
                SkillGap(subject=subject, required_score=required, status="not_assessed", severity="unassessed")
            )
        elif score < required:
            shortfall = round(required - score, 1)
            gaps.append(
                SkillGap(
                    subject=subject,
                    required_score=required,
                    your_score=score,
                    shortfall=shortfall,
                    status="below_benchmark",
                    severity=_severity(shortfall),
                )
            )
        else:
            strengths.append(
                Strength(subject=subject, required_score=required, your_score=score, margin=round(score - required, 1))
            )

    # Real gaps first (largest shortfall first), then unassessed subjects.
    gaps.sort(key=lambda g: (g.shortfall is None, -(g.shortfall or 0), -g.required_score))
    strengths.sort(key=lambda s: s.margin)

    ignored = sorted(name for name in subject_scores if _norm_subject(name) not in benchmark_keys)
    readiness = round(100 * len(strengths) / len(benchmarks))
    return gaps, strengths, readiness, ignored


def _foundation_action(gap: SkillGap) -> str:
    if gap.status == "not_assessed":
        return f"{gap.subject}: take a timed diagnostic test to establish your baseline, then list the weakest topics."
    return (
        f"{gap.subject}: close the {gap.shortfall:.0f}-mark gap. Revise core concepts, list weak topics "
        "from past tests and solve foundational problems."
    )


def build_recovery_plan(career_goal: str, gaps: list[SkillGap], strengths: list[Strength], study_hours: float) -> list[WeekPlan]:
    """Generate a 4-week plan from the prioritised skill gaps."""
    hours = int(min(12, max(3, round(0.4 * study_hours))))  # share of weekly study time for career skills
    mock_target = "; ".join(f"{g.subject} >= {g.required_score:.0f}" for g in gaps)

    # ---- No gaps: stretch plan -------------------------------------------
    if not gaps:
        thin = [s.subject for s in strengths[:2]]
        project = ROLE_PROJECTS[career_goal]
        return [
            WeekPlan(
                week=1,
                theme="Widen your margin",
                focus_subjects=thin,
                actions=[f"{s}: revisit harder topics so your margin above the benchmark grows." for s in thin],
                milestone="Hardest topics in your thinnest subjects revised.",
                suggested_hours=hours,
            ),
            WeekPlan(
                week=2,
                theme="Build something real",
                focus_subjects=[s.subject for s in strengths],
                actions=[f"Start {project}.", "Apply at least two benchmark subjects directly in the project."],
                milestone="Working first version of the project.",
                suggested_hours=hours,
            ),
            WeekPlan(
                week=3,
                theme="Finish and document",
                focus_subjects=[s.subject for s in strengths],
                actions=["Complete the project, add a clear README and publish it.", "Note what each subject contributed."],
                milestone="Project published and documented.",
                suggested_hours=hours,
            ),
            WeekPlan(
                week=4,
                theme="Interview and exam readiness",
                focus_subjects=[s.subject for s in strengths],
                actions=["Take a timed mock covering all benchmark subjects.", "Fix the two weakest answers and repeat."],
                milestone="Timed mock passed with margin in every subject.",
                suggested_hours=hours,
            ),
        ]

    names = [g.subject for g in gaps]
    week4 = WeekPlan(
        week=4,
        theme="Mock assessment and consolidation",
        focus_subjects=names,
        actions=[
            f"Take a timed mock covering {', '.join(names)}.",
            "Review every mistake in your error log and re-solve those questions.",
            "Re-check each score against its benchmark and note what still lags.",
        ],
        milestone=f"Timed mock at or above benchmark: {mock_target}.",
        suggested_hours=hours,
    )

    # ---- 1-2 gaps: phased plan on all gap subjects ------------------------
    if len(gaps) <= 2:
        return [
            WeekPlan(
                week=1,
                theme="Diagnose and rebuild fundamentals",
                focus_subjects=names,
                actions=[_foundation_action(g) for g in gaps],
                milestone="Weak topics listed and core concepts revised for every focus subject.",
                suggested_hours=hours,
            ),
            WeekPlan(
                week=2,
                theme="Guided practice",
                focus_subjects=names,
                actions=[f"{g.subject}: practise topic-wise questions daily and log every mistake with its cause." for g in gaps],
                milestone="Steady accuracy on topic-wise practice sets.",
                suggested_hours=hours,
            ),
            WeekPlan(
                week=3,
                theme=f"Apply it like a {career_goal}",
                focus_subjects=names,
                actions=[f"{g.subject}: {PRACTICE_HINTS[g.subject]}." for g in gaps],
                milestone="One applied exercise completed per focus subject.",
                suggested_hours=hours,
            ),
            week4,
        ]

    # ---- 3+ gaps: one sprint per week, biggest gaps first ------------------
    weeks = []
    for index in range(3):
        chunk = gaps[index::3]
        chunk_names = [g.subject for g in chunk]
        actions = []
        for g in chunk:
            actions.append(_foundation_action(g))
            actions.append(f"{g.subject}: then {PRACTICE_HINTS[g.subject]}.")
        weeks.append(
            WeekPlan(
                week=index + 1,
                theme=f"Sprint: {', '.join(chunk_names)}",
                focus_subjects=chunk_names,
                actions=actions,
                milestone="; ".join(f"{g.subject} topic-wise tests at or above {g.required_score:.0f}" for g in chunk),
                suggested_hours=hours,
            )
        )
    weeks.append(week4)
    return weeks


def explain_prediction(pipeline, means: dict[str, float], features: dict) -> Explanation:
    """
    Per-student feature impact based on deviation from the cohort mean.

    For each feature we replace the student's value with the cohort mean and
    measure how the prediction changes: impact = f(student) - f(student with
    feature at mean). Positive impact = this feature is lifting the score.
    Impacts are not strictly additive because the model is non-linear.
    """
    rows = [features]
    for name in FEATURES:
        altered = dict(features)
        altered[name] = float(means[name])
        rows.append(altered)
    preds = _predict_raw(pipeline, rows)

    drivers: list[Driver] = []
    for i, name in enumerate(FEATURES):
        impact = float(preds[0] - preds[i + 1])
        label, unit = FEATURE_META[name]
        value, mean = float(features[name]), float(means[name])
        direction = "above" if value >= mean else "below"
        effect = "lifting" if impact > 0 else "lowering"
        explanation = (
            f"{label} ({value:g}{unit}) is {abs(value - mean):.1f}{unit} {direction} the cohort average "
            f"of {mean:.1f}{unit}, {effect} the prediction by about {abs(impact):.2f} CGPA."
        )
        drivers.append(
            Driver(
                feature=name,
                label=label,
                student_value=round(value, 2),
                cohort_mean=round(mean, 2),
                impact_cgpa=round(impact, 3),
                explanation=explanation,
            )
        )

    positives = sorted((d for d in drivers if d.impact_cgpa >= MIN_DRIVER_IMPACT), key=lambda d: -d.impact_cgpa)[:2]
    risks = sorted((d for d in drivers if d.impact_cgpa <= -MIN_DRIVER_IMPACT), key=lambda d: d.impact_cgpa)[:2]
    return Explanation(positive_drivers=positives, risk_drivers=risks)


def range_warnings(values: dict[str, float]) -> list[str]:
    warnings = []
    for name, value in values.items():
        low, high = TRAINING_RANGES[name]
        if value < low or value > high:
            label = FEATURE_META[name][0]
            warnings.append(
                f"{label} of {value:g} is outside the range the model was trained on ({low}-{high}); "
                "the prediction may be less reliable."
            )
    return warnings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "pipeline"),
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
    }


@app.get("/api/careers")
def careers():
    """Roles and benchmarks, so the UI can populate its dropdown."""
    return CAREER_MATRIX


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(
    student: StudentRequest,
    pipeline=Depends(get_pipeline),
    means: dict = Depends(get_feature_means),
):
    features = {name: float(getattr(student, name)) for name in FEATURES}

    predicted = predict_cgpa(pipeline, features)
    gap = compute_academic_gap(student.target_cgpa, predicted, student.midterm_score)
    gaps, strengths, readiness, ignored = analyze_skill_gaps(student.career_goal, student.subject_scores)
    plan = build_recovery_plan(student.career_goal, gaps, strengths, student.study_hours_weekly)
    explanation = explain_prediction(pipeline, means, features)

    warnings = range_warnings(features)
    if ignored:
        warnings.append(
            f"Subjects not used for {student.career_goal} (check spelling): {', '.join(ignored)}."
        )

    return AnalysisResponse(
        predicted_cgpa=predicted,
        risk_tier=assign_risk_tier(predicted),
        academic_gap=gap,
        career_goal=student.career_goal,
        career_readiness_pct=readiness,
        skill_gaps=gaps,
        strengths=strengths,
        recovery_plan=plan,
        explanation=explanation,
        warnings=warnings,
    )


@app.post("/api/simulate", response_model=SimulationResponse)
def simulate(sim: SimulationRequest, pipeline=Depends(get_pipeline)):
    base = {
        "grade_10": sim.grade_10,
        "grade_12": sim.grade_12,
        "current_cgpa": sim.current_cgpa,
        "attendance": sim.attendance,
        "midterm_score": sim.midterm_score,
        "study_hours_weekly": sim.study_hours_weekly,
    }
    scenario = dict(base)
    if sim.new_attendance is not None:
        scenario["attendance"] = sim.new_attendance
    if sim.new_study_hours_weekly is not None:
        scenario["study_hours_weekly"] = sim.new_study_hours_weekly

    baseline = predict_cgpa(pipeline, {k: float(v) for k, v in base.items()})
    simulated = predict_cgpa(pipeline, {k: float(v) for k, v in scenario.items()})

    gap = None
    if sim.target_cgpa is not None:
        gap = compute_academic_gap(sim.target_cgpa, simulated, sim.midterm_score)

    return SimulationResponse(
        baseline_cgpa=baseline,
        simulated_cgpa=simulated,
        cgpa_change=round(simulated - baseline, 2),
        baseline_risk_tier=assign_risk_tier(baseline),
        risk_tier=assign_risk_tier(simulated),
        scenario=SimulationScenario(
            attendance=scenario["attendance"], study_hours_weekly=scenario["study_hours_weekly"]
        ),
        academic_gap=gap,
        warnings=range_warnings({k: float(v) for k, v in scenario.items()}),
    )
