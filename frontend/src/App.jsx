import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  BookOpen,
  Briefcase,
  Check,
  Clock,
  Flag,
  FlaskConical,
  GraduationCap,
  Info,
  ListChecks,
  Loader,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react";

/* ==========================================================================
   Configuration
   ========================================================================== */
const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000/api").replace(/\/$/, "");

// Mirrors the backend's CAREER_MATRIX; replaced by GET /api/careers when the API is reachable.
const FALLBACK_CAREERS = {
  "Software Engineer": { "Data Structures": 70, Algorithms: 70, DBMS: 65, "Operating Systems": 60 },
  "Data Analyst": { Statistics: 70, DBMS: 70, Mathematics: 65, "Python Programming": 65 },
  "Web Developer": { "Web Technologies": 70, DBMS: 60, "Data Structures": 60, "Computer Networks": 55 },
  "AI/ML Engineer": {
    Mathematics: 75,
    Statistics: 70,
    "Python Programming": 70,
    "Machine Learning": 70,
    "Data Structures": 70,
  },
};

// The three always-visible subject sliders. Other subjects of the chosen role are optional inputs.
const CORE_SUBJECTS = ["Data Structures", "DBMS", "Mathematics"];

const FIELDS = [
  { key: "grade_10", label: "10th Board %", min: 0, max: 100, step: 0.1, suffix: "%" },
  { key: "grade_12", label: "12th Board %", min: 0, max: 100, step: 0.1, suffix: "%" },
  { key: "current_cgpa", label: "Current CGPA", min: 0, max: 10, step: 0.01, suffix: "/10" },
  { key: "midterm_score", label: "Midterm Avg", min: 0, max: 100, step: 0.1, suffix: "/100" },
  { key: "attendance", label: "Attendance %", min: 0, max: 100, step: 0.1, suffix: "%" },
  { key: "study_hours_weekly", label: "Study Hours / Week", min: 0, max: 168, step: 0.5, suffix: "hrs" },
];
const TARGET_FIELD = { key: "target_cgpa", label: "Target CGPA", min: 0, max: 10, step: 0.01, suffix: "/10" };

const EMPTY_FORM = {
  grade_10: "",
  grade_12: "",
  current_cgpa: "",
  midterm_score: "",
  attendance: "",
  study_hours_weekly: "",
  target_cgpa: "",
  career_goal: "Software Engineer",
  core: { "Data Structures": 50, DBMS: 50, Mathematics: 50 },
  extras: {},
};

const SAMPLE_FORM = {
  grade_10: "78",
  grade_12: "74",
  current_cgpa: "6.8",
  midterm_score: "58",
  attendance: "72",
  study_hours_weekly: "10",
  target_cgpa: "8.0",
  career_goal: "Software Engineer",
  core: { "Data Structures": 62, DBMS: 70, Mathematics: 55 },
  extras: { Algorithms: "68", "Operating Systems": "52" },
};

/* ==========================================================================
   API helpers
   ========================================================================== */
function formatDetail(detail) {
  if (!detail) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        return field ? `${field}: ${d.msg}` : d.msg;
      })
      .join(" · ");
  }
  return null;
}

async function apiFetch(path, { method = "GET", body, signal } = {}) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method,
      signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    const networkError = new Error(`Cannot reach the EduPulse API at ${API_URL}. Is the backend running?`);
    networkError.network = true;
    throw networkError;
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      message = formatDetail(data.detail) ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(message);
  }
  return res.json();
}

/* ==========================================================================
   Hooks
   ========================================================================== */
function useApiStatus() {
  const [status, setStatus] = useState("checking"); // checking | online | offline
  const check = useCallback(async () => {
    try {
      await apiFetch("/health", { signal: AbortSignal.timeout(4000) });
      setStatus("online");
    } catch {
      setStatus("offline");
    }
  }, []);
  useEffect(() => {
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, [check]);
  return { status, setStatus, check };
}

function useDebounced(value, delay) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

/* ==========================================================================
   Form helpers
   ========================================================================== */
function validate(form) {
  const errors = {};
  for (const field of [...FIELDS, TARGET_FIELD]) {
    const raw = form[field.key];
    const n = Number(raw);
    if (raw === "" || Number.isNaN(n)) errors[field.key] = "Required";
    else if (n < field.min || n > field.max) errors[field.key] = `Enter ${field.min}–${field.max}`;
  }
  for (const [subject, raw] of Object.entries(form.extras)) {
    if (raw === "") continue;
    const n = Number(raw);
    if (Number.isNaN(n) || n < 0 || n > 100) errors[`extra:${subject}`] = "Enter 0–100";
  }
  return errors;
}

function buildPayload(form, careers) {
  const roleSubjects = careers[form.career_goal] ?? {};
  const subject_scores = {};
  // Only send subjects the chosen role is benchmarked on; the API warns about the rest.
  for (const subject of CORE_SUBJECTS) {
    if (subject in roleSubjects) subject_scores[subject] = Number(form.core[subject]);
  }
  for (const [subject, raw] of Object.entries(form.extras)) {
    if (subject in roleSubjects && raw !== "") subject_scores[subject] = Number(raw);
  }
  return {
    grade_10: Number(form.grade_10),
    grade_12: Number(form.grade_12),
    current_cgpa: Number(form.current_cgpa),
    attendance: Number(form.attendance),
    midterm_score: Number(form.midterm_score),
    study_hours_weekly: Number(form.study_hours_weekly),
    target_cgpa: Number(form.target_cgpa),
    career_goal: form.career_goal,
    subject_scores,
  };
}

/* ==========================================================================
   Theming helpers
   ========================================================================== */
function tierTheme(tier = "") {
  if (tier.startsWith("Low")) {
    return {
      Icon: ShieldCheck,
      card: "border-emerald-200 bg-emerald-50",
      number: "text-emerald-600",
      badge: "bg-emerald-100 text-emerald-800 ring-emerald-300",
    };
  }
  if (tier.startsWith("Moderate")) {
    return {
      Icon: ShieldAlert,
      card: "border-amber-200 bg-amber-50",
      number: "text-amber-600",
      badge: "bg-amber-100 text-amber-800 ring-amber-300",
    };
  }
  return {
    Icon: ShieldAlert,
    card: "border-red-200 bg-red-50",
    number: "text-red-600",
    badge: "bg-red-100 text-red-800 ring-red-300",
  };
}

const SEVERITY_STYLES = {
  critical: { badge: "bg-red-100 text-red-700 ring-red-200", bar: "bg-red-500", label: "Critical" },
  high: { badge: "bg-orange-100 text-orange-700 ring-orange-200", bar: "bg-orange-500", label: "High" },
  moderate: { badge: "bg-amber-100 text-amber-700 ring-amber-200", bar: "bg-amber-500", label: "Moderate" },
  unassessed: { badge: "bg-slate-100 text-slate-600 ring-slate-200", bar: "bg-slate-300", label: "Not assessed" },
};

const fmt2 = (n) => Number(n).toFixed(2);
const signed = (n, digits = 2) => `${n > 0 ? "+" : ""}${Number(n).toFixed(digits)}`;

/* ==========================================================================
   Small UI building blocks
   ========================================================================== */
function Spinner({ className = "" }) {
  return <Loader className={`animate-spin ${className}`} aria-hidden="true" />;
}

function Card({ title, subtitle, icon: Icon, action, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6 ${className}`}>
      {(title || action) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            {Icon && (
              <div className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-indigo-50 text-indigo-600">
                <Icon size={17} />
              </div>
            )}
            <div>
              <h2 className="text-base font-semibold text-slate-900">{title}</h2>
              {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
            </div>
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

function StatusBadge({ status, onClick }) {
  const styles = {
    online: { cls: "bg-emerald-50 text-emerald-700 ring-emerald-200", label: "API Connected", dot: "bg-emerald-500" },
    offline: { cls: "bg-red-50 text-red-700 ring-red-200", label: "API Offline", dot: "bg-red-500" },
    checking: { cls: "bg-slate-100 text-slate-600 ring-slate-200", label: "Checking…", dot: "bg-slate-400" },
  }[status];
  const Icon = status === "offline" ? WifiOff : status === "online" ? Wifi : null;
  return (
    <button
      type="button"
      onClick={onClick}
      title={`${API_URL} · click to re-check`}
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium ring-1 ring-inset transition hover:brightness-95 ${styles.cls}`}
    >
      {Icon ? (
        <Icon size={14} />
      ) : (
        <span className="relative flex h-2 w-2">
          <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${styles.dot}`} />
          <span className={`relative inline-flex h-2 w-2 rounded-full ${styles.dot}`} />
        </span>
      )}
      <span className="hidden sm:inline">{styles.label}</span>
      <span className="sm:hidden">{status === "online" ? "Online" : status === "offline" ? "Offline" : "…"}</span>
    </button>
  );
}

function NumberField({ field, value, error, onChange }) {
  const id = `field-${field.key.replace(/\s+/g, "-")}`;
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-xs font-medium text-slate-600">
        {field.label}
      </label>
      <div className="relative">
        <input
          id={id}
          type="number"
          inputMode="decimal"
          min={field.min}
          max={field.max}
          step={field.step}
          value={value}
          onChange={(e) => onChange(field.key, e.target.value)}
          aria-invalid={Boolean(error)}
          className={`w-full rounded-lg border bg-white py-2 pl-3 pr-11 text-sm text-slate-900 outline-none transition focus:ring-2 ${
            error
              ? "border-red-400 focus:ring-red-200"
              : "border-slate-300 focus:border-indigo-500 focus:ring-indigo-200"
          }`}
        />
        <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-xs text-slate-400">
          {field.suffix}
        </span>
      </div>
      {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
    </div>
  );
}

function SubjectSlider({ subject, value, benchmark, onChange }) {
  const id = `slider-${subject.replace(/\s+/g, "-")}`;
  const inRole = benchmark !== undefined;
  const met = inRole && value >= benchmark;
  return (
    <div className={inRole ? "" : "opacity-60"}>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-sm font-medium text-slate-700">
          {subject}
        </label>
        <span className="text-sm font-semibold tabular-nums text-slate-900">{value}</span>
      </div>
      <input
        id={id}
        type="range"
        min={0}
        max={100}
        step={1}
        value={value}
        onChange={(e) => onChange(subject, Number(e.target.value))}
        className="mt-1 w-full accent-indigo-600"
      />
      <p className={`text-xs ${!inRole ? "text-slate-400" : met ? "text-emerald-600" : "text-amber-600"}`}>
        {inRole ? `Benchmark ≥ ${benchmark}${met ? " · met" : ""}` : "Not benchmarked for this role"}
      </p>
    </div>
  );
}

/* ==========================================================================
   Input form (left column)
   ========================================================================== */
function InputForm({ form, errors, careers, loading, onField, onCore, onExtra, onRole, onSubmit }) {
  const roleSubjects = careers[form.career_goal] ?? {};
  const extraSubjects = Object.keys(roleSubjects).filter((s) => !CORE_SUBJECTS.includes(s));

  return (
    <form
      onSubmit={onSubmit}
      noValidate
      className="space-y-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6"
    >
      <div>
        <h2 className="flex items-center gap-2 text-base font-semibold text-slate-900">
          <BookOpen size={18} className="text-indigo-600" /> Academic Profile
        </h2>
        <p className="mt-1 text-sm text-slate-500">Enter your current standing to get a full diagnosis.</p>
      </div>

      <fieldset className="space-y-3">
        <legend className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Performance</legend>
        <div className="grid grid-cols-2 gap-3">
          {FIELDS.map((field) => (
            <NumberField
              key={field.key}
              field={field}
              value={form[field.key]}
              error={errors[field.key]}
              onChange={onField}
            />
          ))}
        </div>
      </fieldset>

      <fieldset className="space-y-3">
        <legend className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Goals</legend>
        <div>
          <label htmlFor="career_goal" className="mb-1 block text-xs font-medium text-slate-600">
            Career Goal
          </label>
          <div className="relative">
            <Briefcase size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <select
              id="career_goal"
              value={form.career_goal}
              onChange={(e) => onRole(e.target.value)}
              className="w-full appearance-none rounded-lg border border-slate-300 bg-white py-2 pl-9 pr-3 text-sm text-slate-900 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
            >
              {Object.keys(careers).map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
          </div>
        </div>
        <NumberField field={TARGET_FIELD} value={form.target_cgpa} error={errors.target_cgpa} onChange={onField} />
      </fieldset>

      <fieldset className="space-y-4">
        <legend className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Subject marks</legend>
        {CORE_SUBJECTS.map((subject) => (
          <SubjectSlider
            key={subject}
            subject={subject}
            value={form.core[subject]}
            benchmark={roleSubjects[subject]}
            onChange={onCore}
          />
        ))}

        {extraSubjects.length > 0 && (
          <div className="rounded-xl bg-slate-50 p-3">
            <p className="mb-2 text-xs font-medium text-slate-600">
              Other {form.career_goal} subjects <span className="font-normal text-slate-400">(optional)</span>
            </p>
            <div className="grid grid-cols-2 gap-3">
              {extraSubjects.map((subject) => (
                <NumberField
                  key={subject}
                  field={{
                    key: `extra-${subject}`,
                    label: `${subject} (≥ ${roleSubjects[subject]})`,
                    min: 0,
                    max: 100,
                    step: 1,
                    suffix: "/100",
                  }}
                  value={form.extras[subject] ?? ""}
                  error={errors[`extra:${subject}`]}
                  onChange={(_, value) => onExtra(subject, value)}
                />
              ))}
            </div>
            <p className="mt-2 text-xs text-slate-500">Leave blank to mark a subject as not assessed.</p>
          </div>
        )}
      </fieldset>

      <button
        type="submit"
        disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70"
      >
        {loading ? <Spinner className="h-4 w-4" /> : <Activity size={16} />}
        {loading ? "Running diagnosis…" : "Run Comprehensive Diagnosis"}
      </button>
    </form>
  );
}

/* ==========================================================================
   Results (right column)
   ========================================================================== */
function VitalMetrics({ result }) {
  const theme = tierTheme(result.risk_tier);
  const gap = result.academic_gap;
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      <div className={`rounded-2xl border p-5 ${theme.card}`}>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Predicted CGPA</p>
        <p className={`mt-2 text-5xl font-bold tabular-nums ${theme.number}`}>{fmt2(result.predicted_cgpa)}</p>
        <p className="mt-1 text-sm text-slate-500">projected final, out of 10</p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Target CGPA</p>
        <p className="mt-2 text-5xl font-bold tabular-nums text-slate-900">{fmt2(gap.target_cgpa)}</p>
        <p className="mt-1 text-sm">
          {gap.on_track ? (
            <span className="font-medium text-emerald-600">On track · {fmt2(Math.abs(gap.cgpa_gap))} margin</span>
          ) : (
            <span className="font-medium text-amber-600">{fmt2(gap.cgpa_gap)} points to close</span>
          )}
        </p>
      </div>

      <div className={`rounded-2xl border p-5 ${theme.card}`}>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Risk Tier</p>
        <div className="mt-3">
          <span
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-semibold ring-1 ring-inset ${theme.badge}`}
          >
            <theme.Icon size={16} />
            {result.risk_tier}
          </span>
        </div>
        <p className="mt-3 text-sm text-slate-500">Based on projected CGPA thresholds (7.5 and 6.0).</p>
      </div>
    </div>
  );
}

function TargetBridge({ gap }) {
  if (gap.on_track) {
    return (
      <div className="flex items-start gap-3 rounded-xl bg-emerald-50 p-4 text-emerald-800">
        <ShieldCheck size={20} className="mt-0.5 shrink-0" />
        <p className="text-sm">{gap.message}</p>
      </div>
    );
  }

  const required = gap.required_final_score;
  const baseline = gap.baseline_exam_score;
  const fillTo = Math.min(100, required);
  const achievable = gap.achievable;

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-slate-500">Required score in remaining finals</p>
          <p className={`text-5xl font-bold tabular-nums ${achievable ? "text-indigo-600" : "text-red-600"}`}>
            {required.toFixed(1)}
            <span className="ml-1 text-xl font-medium text-slate-400">/ 100</span>
          </p>
        </div>
        <div className="text-right text-sm text-slate-500">
          <p>
            Gap to close <span className="font-semibold text-slate-900">+{fmt2(gap.cgpa_gap)}</span> CGPA
          </p>
          <p>
            Extra marks over midterm{" "}
            <span className="font-semibold text-slate-900">{signed(required - baseline, 1)}</span>
          </p>
        </div>
      </div>

      <div className="relative mt-5 h-3 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
        <div className="absolute inset-y-0 left-0 bg-slate-300" style={{ width: `${baseline}%` }} />
        <div
          className={`absolute inset-y-0 ${achievable ? "bg-indigo-500" : "bg-red-500"}`}
          style={{ left: `${baseline}%`, width: `${Math.max(0, fillTo - baseline)}%` }}
        />
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-500">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-slate-300" /> Midterm baseline {baseline.toFixed(0)}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className={`h-2 w-2 rounded-full ${achievable ? "bg-indigo-500" : "bg-red-500"}`} /> Extra marks needed
        </span>
      </div>

      <p className={`mt-4 rounded-xl p-3 text-sm ${achievable ? "bg-indigo-50 text-indigo-900" : "bg-red-50 text-red-800"}`}>
        {gap.message}
      </p>
      <p className="mt-3 flex items-start gap-2 text-xs text-slate-400">
        <Info size={14} className="mt-0.5 shrink-0" />
        Estimate assumes finals make up {Math.round(gap.finals_weight * 100)}% of the final CGPA and that the model
        expects finals to land near your midterm score. Adjust to your college's actual weighting.
      </p>
    </div>
  );
}

function DriverCard({ driver, tone, maxImpact }) {
  const positive = tone === "positive";
  const width = maxImpact > 0 ? Math.max(8, (Math.abs(driver.impact_cgpa) / maxImpact) * 100) : 8;
  return (
    <div
      className={`rounded-xl border p-4 ${positive ? "border-emerald-200 bg-emerald-50" : "border-red-200 bg-red-50"}`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {positive ? (
            <TrendingUp size={18} className="text-emerald-600" />
          ) : (
            <TrendingDown size={18} className="text-red-600" />
          )}
          <h3 className="text-sm font-semibold text-slate-900">{driver.label}</h3>
        </div>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-semibold tabular-nums ${
            positive ? "bg-emerald-100 text-emerald-800" : "bg-red-100 text-red-800"
          }`}
        >
          {signed(driver.impact_cgpa)} CGPA
        </span>
      </div>
      <div className="mt-3 h-1.5 rounded-full bg-white/70">
        <div
          className={`h-full rounded-full ${positive ? "bg-emerald-500" : "bg-red-500"}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <p className="mt-3 text-sm leading-relaxed text-slate-600">{driver.explanation}</p>
    </div>
  );
}

function Explainability({ explanation }) {
  const { positive_drivers: boosters, risk_drivers: risks } = explanation;
  const maxImpact = Math.max(0, ...[...boosters, ...risks].map((d) => Math.abs(d.impact_cgpa)));
  return (
    <div className="grid gap-5 md:grid-cols-2">
      <div>
        <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-emerald-700">
          <TrendingUp size={16} /> Score boosters
        </h3>
        <div className="space-y-3">
          {boosters.length ? (
            boosters.map((d) => <DriverCard key={d.feature} driver={d} tone="positive" maxImpact={maxImpact} />)
          ) : (
            <p className="rounded-xl border border-dashed border-slate-200 p-4 text-sm text-slate-500">
              No factor is lifting your prediction meaningfully above the cohort average.
            </p>
          )}
        </div>
      </div>
      <div>
        <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-red-700">
          <TrendingDown size={16} /> Risk drivers
        </h3>
        <div className="space-y-3">
          {risks.length ? (
            risks.map((d) => <DriverCard key={d.feature} driver={d} tone="risk" maxImpact={maxImpact} />)
          ) : (
            <p className="rounded-xl border border-dashed border-slate-200 p-4 text-sm text-slate-500">
              Nothing is dragging your prediction down meaningfully.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function SkillGapCard({ gap }) {
  const style = SEVERITY_STYLES[gap.severity] ?? SEVERITY_STYLES.unassessed;
  const assessed = gap.your_score !== null && gap.your_score !== undefined;
  return (
    <div className={`rounded-xl border p-4 ${assessed ? "border-slate-200 bg-white" : "border-dashed border-slate-300 bg-slate-50"}`}>
      <div className="flex items-start justify-between gap-2">
        <h4 className="text-sm font-semibold text-slate-900">{gap.subject}</h4>
        <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${style.badge}`}>
          {style.label}
        </span>
      </div>
      {assessed ? (
        <>
          <div className="relative mt-4 h-2 rounded-full bg-slate-100">
            <div className={`h-full rounded-full ${style.bar}`} style={{ width: `${gap.your_score}%` }} />
            <div
              className="absolute -top-1 h-4 w-0.5 rounded bg-slate-800"
              style={{ left: `${gap.required_score}%` }}
              title={`Benchmark ${gap.required_score}`}
            />
          </div>
          <div className="mt-2 flex justify-between text-xs text-slate-500">
            <span>
              You: <b className="text-slate-800">{gap.your_score}</b>
            </span>
            <span>
              Need: <b className="text-slate-800">{gap.required_score}</b> ·{" "}
              <b className="text-red-600">−{gap.shortfall}</b>
            </span>
          </div>
        </>
      ) : (
        <p className="mt-3 text-xs text-slate-500">
          Benchmark ≥ {gap.required_score}. Add a score for this subject for a sharper diagnosis.
        </p>
      )}
    </div>
  );
}

function RecoveryChecklist({ plan, checked, onToggle }) {
  const total = plan.reduce((sum, w) => sum + w.actions.length, 0);
  const done = Object.values(checked).filter(Boolean).length;
  const pct = total ? Math.round((done / total) * 100) : 0;

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900">
          <ListChecks size={16} className="text-indigo-600" /> 4-week recovery roadmap
        </h3>
        <span className="text-xs font-medium text-slate-500">
          {done}/{total} done · {pct}%
        </span>
      </div>
      <div className="mb-4 h-2 rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-indigo-500 transition-all" style={{ width: `${pct}%` }} />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {plan.map((week) => {
          const weekDone = week.actions.filter((_, i) => checked[`${week.week}:${i}`]).length;
          return (
            <div key={week.week} className="rounded-xl border border-slate-200 p-4">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <span className="rounded-md bg-indigo-50 px-2 py-0.5 text-xs font-semibold text-indigo-700">
                    Week {week.week}
                  </span>
                  <h4 className="mt-2 text-sm font-semibold text-slate-900">{week.theme}</h4>
                </div>
                <span className="inline-flex shrink-0 items-center gap-1 text-xs text-slate-500">
                  <Clock size={12} /> ~{week.suggested_hours} hrs
                </span>
              </div>
              <ul className="mt-3 space-y-2">
                {week.actions.map((action, i) => {
                  const key = `${week.week}:${i}`;
                  const id = `task-${week.week}-${i}`;
                  return (
                    <li key={key}>
                      <label htmlFor={id} className="flex cursor-pointer items-start gap-2.5 text-sm">
                        <input
                          id={id}
                          type="checkbox"
                          checked={Boolean(checked[key])}
                          onChange={() => onToggle(key)}
                          className="mt-0.5 h-4 w-4 shrink-0 rounded border-slate-300 accent-indigo-600"
                        />
                        <span className={checked[key] ? "text-slate-400 line-through" : "text-slate-700"}>{action}</span>
                      </label>
                    </li>
                  );
                })}
              </ul>
              <p className="mt-3 flex items-start gap-1.5 border-t border-slate-100 pt-3 text-xs text-slate-500">
                <Flag size={12} className="mt-0.5 shrink-0 text-indigo-500" />
                <span>
                  <b className="text-slate-600">Milestone:</b> {week.milestone}
                  {weekDone === week.actions.length && <Check size={12} className="ml-1 inline text-emerald-600" />}
                </span>
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SkillGapMatrix({ result, checked, onToggle }) {
  const { skill_gaps: gaps, strengths } = result;
  const readiness = result.career_readiness_pct;
  const readinessColor = readiness >= 75 ? "bg-emerald-500" : readiness >= 40 ? "bg-amber-500" : "bg-red-500";

  return (
    <div className="space-y-6">
      <div>
        <div className="mb-1.5 flex items-center justify-between text-sm">
          <span className="font-medium text-slate-700">
            {result.career_goal} readiness <span className="font-normal text-slate-400">(benchmarks met)</span>
          </span>
          <span className="font-semibold tabular-nums text-slate-900">{readiness}%</span>
        </div>
        <div className="h-2.5 rounded-full bg-slate-100">
          <div className={`h-full rounded-full transition-all ${readinessColor}`} style={{ width: `${readiness}%` }} />
        </div>
      </div>

      {gaps.length > 0 ? (
        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-900">Skill gaps</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {gaps.map((g) => (
              <SkillGapCard key={g.subject} gap={g} />
            ))}
          </div>
        </div>
      ) : (
        <p className="rounded-xl bg-emerald-50 p-4 text-sm text-emerald-800">
          You meet every benchmark for this role. The plan below focuses on widening your margin.
        </p>
      )}

      {strengths.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-900">Benchmarks met</h3>
          <div className="flex flex-wrap gap-2">
            {strengths.map((s) => (
              <span
                key={s.subject}
                className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-800 ring-1 ring-inset ring-emerald-200"
              >
                <Check size={12} /> {s.subject} · {s.your_score}
                <span className="text-emerald-600">({signed(s.margin, 0)})</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <RecoveryChecklist plan={result.recovery_plan} checked={checked} onToggle={onToggle} />
    </div>
  );
}

/* ==========================================================================
   What-If simulator
   ========================================================================== */
function WhatIfSimulator({ payload, result }) {
  const attMin = Math.floor(Math.min(45, payload.attendance));
  const hrsMax = Math.ceil(Math.max(35, payload.study_hours_weekly));
  const hrsMin = Math.floor(Math.min(1, payload.study_hours_weekly));

  const [attendance, setAttendance] = useState(payload.attendance);
  const [hours, setHours] = useState(payload.study_hours_weekly);
  const [sim, setSim] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const debouncedAttendance = useDebounced(attendance, 350);
  const debouncedHours = useDebounced(hours, 350);

  useEffect(() => {
    if (debouncedAttendance === payload.attendance && debouncedHours === payload.study_hours_weekly) {
      setSim(null);
      setError(null);
      setLoading(false);
      return undefined;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    apiFetch("/simulate", {
      method: "POST",
      signal: controller.signal,
      body: {
        grade_10: payload.grade_10,
        grade_12: payload.grade_12,
        current_cgpa: payload.current_cgpa,
        midterm_score: payload.midterm_score,
        attendance: payload.attendance,
        study_hours_weekly: payload.study_hours_weekly,
        new_attendance: debouncedAttendance,
        new_study_hours_weekly: debouncedHours,
        target_cgpa: payload.target_cgpa,
      },
    })
      .then((data) => {
        setSim(data);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setError(err.message);
        setLoading(false);
      });
    return () => controller.abort();
  }, [debouncedAttendance, debouncedHours, payload]);

  const view = sim ?? {
    simulated_cgpa: result.predicted_cgpa,
    cgpa_change: 0,
    risk_tier: result.risk_tier,
    academic_gap: result.academic_gap,
    warnings: [],
  };
  const theme = tierTheme(view.risk_tier);
  const pending = loading || attendance !== debouncedAttendance || hours !== debouncedHours;
  const changed = view.cgpa_change !== 0;
  const tierChanged = view.risk_tier !== result.risk_tier;
  const gap = view.academic_gap;
  const atBaseline = attendance === payload.attendance && hours === payload.study_hours_weekly;

  const reset = () => {
    setAttendance(payload.attendance);
    setHours(payload.study_hours_weekly);
  };

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <div className="space-y-6 lg:col-span-3">
        <div>
          <div className="flex items-baseline justify-between">
            <label htmlFor="sim-attendance" className="text-sm font-medium text-slate-700">
              Attendance
            </label>
            <span className="text-sm tabular-nums text-slate-500">
              <b className="text-lg text-slate-900">{attendance}%</b> · now {payload.attendance}%
            </span>
          </div>
          <input
            id="sim-attendance"
            type="range"
            min={attMin}
            max={100}
            step={1}
            value={attendance}
            onChange={(e) => setAttendance(Number(e.target.value))}
            className="mt-2 w-full accent-indigo-600"
          />
        </div>

        <div>
          <div className="flex items-baseline justify-between">
            <label htmlFor="sim-hours" className="text-sm font-medium text-slate-700">
              Study hours per week
            </label>
            <span className="text-sm tabular-nums text-slate-500">
              <b className="text-lg text-slate-900">{hours} hrs</b> · now {payload.study_hours_weekly} hrs
            </span>
          </div>
          <input
            id="sim-hours"
            type="range"
            min={hrsMin}
            max={hrsMax}
            step={0.5}
            value={hours}
            onChange={(e) => setHours(Number(e.target.value))}
            className="mt-2 w-full accent-indigo-600"
          />
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={reset}
            disabled={atBaseline}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RotateCcw size={14} /> Reset to current
          </button>
          <span className="text-xs text-slate-400">
            Current CGPA and midterm stay fixed, so this shows the direct effect of your habits only.
          </span>
        </div>

        {error && <p className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}
        {view.warnings?.length > 0 && (
          <p className="rounded-lg bg-amber-50 p-3 text-xs text-amber-800">{view.warnings.join(" ")}</p>
        )}
      </div>

      <div className={`rounded-2xl border p-5 lg:col-span-2 ${theme.card}`}>
        <div className="flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Simulated CGPA</p>
          {pending && <Spinner className="h-4 w-4 text-slate-400" />}
        </div>
        <div className="mt-2 flex items-end gap-3">
          <p className={`text-5xl font-bold tabular-nums transition-opacity ${theme.number} ${pending ? "opacity-60" : ""}`}>
            {fmt2(view.simulated_cgpa)}
          </p>
          {changed && (
            <span
              className={`mb-1.5 rounded-full px-2 py-0.5 text-sm font-semibold tabular-nums ${
                view.cgpa_change > 0 ? "bg-emerald-100 text-emerald-800" : "bg-red-100 text-red-800"
              }`}
            >
              {signed(view.cgpa_change)}
            </span>
          )}
        </div>
        <p className="mt-1 text-xs text-slate-500">vs. {fmt2(result.predicted_cgpa)} predicted today</p>

        <div className="mt-4">
          <span
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-inset ${theme.badge}`}
          >
            <theme.Icon size={14} />
            {view.risk_tier}
          </span>
          {tierChanged && <p className="mt-2 text-xs text-slate-600">Moved from “{result.risk_tier}”.</p>}
        </div>

        {gap && (
          <p className="mt-4 border-t border-black/5 pt-3 text-sm text-slate-700">
            {gap.on_track ? (
              <>At this pace you meet your {fmt2(gap.target_cgpa)} target.</>
            ) : gap.achievable ? (
              <>
                Finals needed for {fmt2(gap.target_cgpa)}:{" "}
                <b className="tabular-nums">{gap.required_final_score.toFixed(1)}/100</b>
              </>
            ) : (
              <>Target still needs more than 100/100 in finals.</>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

/* ==========================================================================
   Placeholder / loading states
   ========================================================================== */
function EmptyState() {
  const steps = [
    { Icon: Zap, title: "Predict", text: "A model projects your final CGPA and risk tier." },
    { Icon: Target, title: "Bridge the gap", text: "See the finals score needed to reach your target." },
    { Icon: Briefcase, title: "Close skill gaps", text: "Compare subjects with your career benchmarks." },
  ];
  return (
    <div className="rounded-2xl border border-dashed border-slate-300 bg-white/60 p-8 text-center sm:p-12">
      <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-indigo-50 text-indigo-600">
        <Activity size={28} />
      </div>
      <h2 className="mt-4 text-lg font-semibold text-slate-900">Your diagnosis will appear here</h2>
      <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
        Fill in your academic profile, or click <b>Load Sample Student Data</b> to try it instantly.
      </p>
      <div className="mx-auto mt-8 grid max-w-2xl gap-4 text-left sm:grid-cols-3">
        {steps.map(({ Icon, title, text }) => (
          <div key={title} className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-100">
            <Icon size={18} className="text-indigo-500" />
            <p className="mt-2 text-sm font-semibold text-slate-900">{title}</p>
            <p className="mt-1 text-xs text-slate-500">{text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="animate-pulse space-y-4" aria-busy="true" aria-label="Running diagnosis">
      <div className="grid gap-4 sm:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-36 rounded-2xl bg-slate-200/70" />
        ))}
      </div>
      <div className="h-48 rounded-2xl bg-slate-200/70" />
      <div className="h-64 rounded-2xl bg-slate-200/70" />
    </div>
  );
}

/* ==========================================================================
   App
   ========================================================================== */
export default function App() {
  const { status, setStatus, check } = useApiStatus();

  const [careers, setCareers] = useState(FALLBACK_CAREERS);
  const careersLoaded = useRef(false);

  const [form, setForm] = useState(EMPTY_FORM);
  const [errors, setErrors] = useState({});

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [payload, setPayload] = useState(null); // snapshot of what was analysed
  const [result, setResult] = useState(null);
  const [runId, setRunId] = useState(0);
  const [checked, setChecked] = useState({});

  const resultsRef = useRef(null);

  // Pull the role/benchmark list from the API once it is reachable.
  useEffect(() => {
    if (status !== "online" || careersLoaded.current) return;
    apiFetch("/careers")
      .then((data) => {
        if (data && typeof data === "object" && Object.keys(data).length) {
          setCareers(data);
          careersLoaded.current = true;
        }
      })
      .catch(() => {});
  }, [status]);

  const setField = (key, value) => {
    setForm((f) => ({ ...f, [key]: value }));
    setErrors((e) => (e[key] ? { ...e, [key]: undefined } : e));
  };
  const setCore = (subject, value) => setForm((f) => ({ ...f, core: { ...f.core, [subject]: value } }));
  const setExtra = (subject, value) => {
    setForm((f) => ({ ...f, extras: { ...f.extras, [subject]: value } }));
    setErrors((e) => (e[`extra:${subject}`] ? { ...e, [`extra:${subject}`]: undefined } : e));
  };
  const setRole = (role) => setForm((f) => ({ ...f, career_goal: role }));

  const loadSample = () => {
    setForm(SAMPLE_FORM);
    setErrors({});
    setError(null);
  };

  const toggleTask = (key) => setChecked((c) => ({ ...c, [key]: !c[key] }));

  const handleSubmit = async (event) => {
    event.preventDefault();
    const found = validate(form);
    setErrors(found);
    if (Object.values(found).some(Boolean)) return;

    const body = buildPayload(form, careers);
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch("/analyze", { method: "POST", body });
      setPayload(body);
      setResult(data);
      setChecked({});
      setRunId((id) => id + 1);
      setStatus("online");
      if (window.innerWidth < 1024) {
        requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      }
    } catch (err) {
      setError(err.message);
      if (err.network) setStatus("offline");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* ------------------------------ Top bar ------------------------------ */}
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-indigo-600 text-white shadow-sm">
              <GraduationCap size={20} />
            </div>
            <div>
              <h1 className="text-lg font-bold leading-none tracking-tight">EduPulse</h1>
              <p className="mt-0.5 hidden text-xs text-slate-500 sm:block">Explainable academic trajectory</p>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <StatusBadge status={status} onClick={check} />
            <button
              type="button"
              onClick={loadSample}
              className="inline-flex items-center gap-2 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-sm font-medium text-indigo-700 transition hover:bg-indigo-100"
            >
              <Sparkles size={15} />
              <span className="hidden sm:inline">Load Sample Student Data</span>
              <span className="sm:hidden">Sample</span>
            </button>
          </div>
        </div>
      </header>

      {/* ------------------------------- Body -------------------------------- */}
      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-12 lg:items-start">
        <div className="lg:col-span-4 lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto lg:rounded-2xl">
          <InputForm
            form={form}
            errors={errors}
            careers={careers}
            loading={loading}
            onField={setField}
            onCore={setCore}
            onExtra={setExtra}
            onRole={setRole}
            onSubmit={handleSubmit}
          />
        </div>

        <div ref={resultsRef} className="scroll-mt-20 space-y-6 lg:col-span-8">
          {error && (
            <div role="alert" className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-red-800">
              <ShieldAlert size={20} className="mt-0.5 shrink-0" />
              <div className="text-sm">
                <p className="font-semibold">Diagnosis failed</p>
                <p className="mt-0.5">{error}</p>
                {status === "offline" && (
                  <p className="mt-1 text-red-700">
                    Start the API with <code className="rounded bg-red-100 px-1">uvicorn main:app --reload --port 8000</code>{" "}
                    inside <code className="rounded bg-red-100 px-1">backend/</code>.
                  </p>
                )}
              </div>
            </div>
          )}

          {loading && !result ? (
            <LoadingSkeleton />
          ) : !result ? (
            <EmptyState />
          ) : (
            <div className={`space-y-6 transition-opacity ${loading ? "opacity-60" : ""}`}>
              {result.warnings?.length > 0 && (
                <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-amber-900">
                  <Info size={18} className="mt-0.5 shrink-0" />
                  <ul className="space-y-1 text-sm">
                    {result.warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              <VitalMetrics result={result} />

              <Card
                title="Academic Target Bridge"
                subtitle="What you need in the remaining finals to reach your target CGPA"
                icon={Target}
              >
                <TargetBridge gap={result.academic_gap} />
              </Card>

              <Card
                title="Why This Score?"
                subtitle="How each factor moves your prediction compared with the average student"
                icon={Sparkles}
              >
                <Explainability explanation={result.explanation} />
              </Card>

              <Card
                title="Career Skill Gap Matrix"
                subtitle={`Your subject marks against ${result.career_goal} benchmarks`}
                icon={Briefcase}
              >
                <SkillGapMatrix result={result} checked={checked} onToggle={toggleTask} />
              </Card>

              <Card
                title="What-If Academic Simulator"
                subtitle="Drag the sliders to see how better habits change your projected CGPA"
                icon={FlaskConical}
              >
                <WhatIfSimulator key={runId} payload={payload} result={result} />
              </Card>

              <p className="pb-4 text-center text-xs text-slate-400">
                Predictions come from a model trained on synthetic data. They illustrate the system and are not
                validated on real students.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
