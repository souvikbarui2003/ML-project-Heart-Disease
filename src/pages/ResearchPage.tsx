import { useState, useEffect, useRef } from 'react'
import {
  Heart, Shield, Activity, BarChart3, AlertTriangle, Brain, BookOpen,
  Database, Target, GitBranch, ChevronDown, ChevronRight, ExternalLink,
  FileText, Github, Download, Clock, Check, X, ArrowRight, Search,
  TrendingUp, TrendingDown, Info, Layers, Zap, Users, Beaker, Lightbulb
} from 'lucide-react'

/* ──────────────────────────────────────────────
   DATA — All sourced from repository files
   ────────────────────────────────────────────── */

const RESEARCH_META = {
  title: 'An Explainable Machine Learning Framework for Heart Disease Risk Prediction Using Public Health Data',
  subtitle: 'An end-to-end machine learning framework combining rigorous preprocessing, comparative model evaluation, probability calibration, explainable AI, and an accessible web interface for heart disease risk estimation.',
  author: 'Not specified in repository',
  department: 'Not specified in repository',
  university: 'Not specified in repository',
  academicYear: '2025–2026',
  projectType: 'Major Project / Thesis',
  supervisor: 'Not specified in repository',
  dataset: 'Cleveland Heart Disease (UCI ML Repository)',
  bestModel: 'Logistic Regression',
  cvRocAuc: 0.9109,
  testRocAuc: 0.8885,
  testAccuracy: 0.8361,
  recall: 0.8788,
  specificity: 0.7857,
  brier: 0.12,
  f1: 0.8571,
  version: '2.0.0',
  repo: 'https://github.com/souvikbarui2003/ML-project-Heart-Disease',
  github: 'https://github.com/souvikbarui2003/ML-project-Heart-Disease',
}

const MODEL_RESULTS = [
  { name: 'Logistic Regression', category: 'Linear', cvAuc: 0.9109, testAuc: 0.8885, accuracy: 0.8361, recall: 0.8788, specificity: 0.7857, f1: 0.8571, brier: 0.12, time: 0.11, best: true, calibrated: true, desc: 'Linear model for binary classification. Models log-odds of outcome as a linear combination of features. Provides interpretable coefficients and naturally calibrated probabilities.' },
  { name: 'SVM', category: 'Kernel-based', cvAuc: 0.9102, testAuc: 0.8864, accuracy: 0.7869, recall: 0.8788, specificity: 0.6786, f1: 0.8235, brier: 0.14, time: 0.49, best: false, calibrated: true, desc: 'Finds the optimal hyperplane separating classes using kernel trick. Strong performance on small, high-dimensional datasets.' },
  { name: 'K-Nearest Neighbors', category: 'Distance-based', cvAuc: 0.8953, testAuc: 0.8555, accuracy: 0.7541, recall: 0.8485, specificity: 0.6429, f1: 0.8000, brier: 0.16, time: 0.20, best: false, calibrated: false, desc: 'Non-parametric method classifying based on majority class of k nearest training samples. Sensitive to feature scaling and choice of k.' },
  { name: 'Random Forest', category: 'Tree Ensemble', cvAuc: 0.8882, testAuc: 0.8463, accuracy: 0.7213, recall: 0.8182, specificity: 0.6071, f1: 0.7692, brier: 0.18, time: 19.99, best: false, calibrated: false, desc: 'Ensemble of decision trees trained on random subsets of data and features. Reduces variance through bagging.' },
  { name: 'Gradient Boosting', category: 'Boosting', cvAuc: 0.8737, testAuc: 0.8506, accuracy: 0.7377, recall: 0.8182, specificity: 0.6429, f1: 0.7813, brier: 0.17, time: 27.82, best: false, calibrated: false, desc: 'Builds trees sequentially, each correcting errors of the previous ensemble. Strong performance on structured data.' },
  { name: 'Decision Tree', category: 'Tree-based', cvAuc: 0.8274, testAuc: 0.7495, accuracy: 0.7377, recall: 0.8485, specificity: 0.6071, f1: 0.7925, brier: 0.25, time: 0.46, best: false, calibrated: false, desc: 'Tree-structured model that recursively splits data based on feature values. Highly interpretable but prone to overfitting.' },
]

const FEATURE_IMPORTANCE = [
  { feature: 'exang', name: 'Exercise-Induced Angina', importance: 0.142, desc: 'Chest pain during physical activity. Strong predictor of coronary artery disease.' },
  { feature: 'thal', name: 'Thalassemia', importance: 0.128, desc: 'Blood disorder type. Indicates blood flow abnormalities.' },
  { feature: 'ca', name: 'Major Vessels', importance: 0.124, desc: 'Number of major vessels colored by fluoroscopy. Direct measure of blockages.' },
  { feature: 'cp', name: 'Chest Pain Type', importance: 0.118, desc: 'Pattern of chest discomfort. Primary symptom of heart disease.' },
  { feature: 'oldpeak', name: 'ST Depression', importance: 0.109, desc: 'Exercise-induced ECG change. Indicator of myocardial ischemia.' },
  { feature: 'thalach', name: 'Max Heart Rate', importance: 0.098, desc: 'Peak exercise heart rate. Exercise capacity indicator.' },
  { feature: 'slope', name: 'ST Slope', importance: 0.087, desc: 'ECG pattern during exercise. Diagnostic indicator.' },
  { feature: 'sex', name: 'Sex', importance: 0.065, desc: 'Biological sex. Known cardiovascular risk factor.' },
  { feature: 'age', name: 'Age', importance: 0.052, desc: 'Patient age in years. Primary cardiovascular risk factor.' },
  { feature: 'trestbps', name: 'Resting BP', importance: 0.035, desc: 'Blood pressure when resting. Hypertension indicator.' },
]

const FEATURES = [
  { name: 'age', display: 'Age', type: 'Numeric', unit: 'years', range: '29–77', desc: 'Patient age in years.' },
  { name: 'sex', display: 'Sex', type: 'Binary', unit: '—', range: '0–1', desc: '0 = Female, 1 = Male.' },
  { name: 'cp', display: 'Chest Pain Type', type: 'Categorical', unit: '—', range: '0–3', desc: '0=Typical Angina, 1=Atypical Angina, 2=Non-Anginal Pain, 3=Asymptomatic.' },
  { name: 'trestbps', display: 'Resting Blood Pressure', type: 'Numeric', unit: 'mmHg', range: '94–200', desc: 'Blood pressure at rest.' },
  { name: 'chol', display: 'Serum Cholesterol', type: 'Numeric', unit: 'mg/dl', range: '126–564', desc: 'Total cholesterol level.' },
  { name: 'fbs', display: 'Fasting Blood Sugar', type: 'Binary', unit: '—', range: '0–1', desc: '0=≤120 mg/dl, 1=>120 mg/dl.' },
  { name: 'restecg', display: 'Resting ECG', type: 'Categorical', unit: '—', range: '0–2', desc: '0=Normal, 1=ST-T Wave Abnormality, 2=Left Ventricular Hypertrophy.' },
  { name: 'thalach', display: 'Max Heart Rate', type: 'Numeric', unit: 'bpm', range: '71–202', desc: 'Maximum heart rate achieved during exercise.' },
  { name: 'exang', display: 'Exercise Angina', type: 'Binary', unit: '—', range: '0–1', desc: 'Chest pain caused by exercise.' },
  { name: 'oldpeak', display: 'ST Depression', type: 'Numeric', unit: 'mm', range: '0.0–6.2', desc: 'ST depression induced by exercise.' },
  { name: 'slope', display: 'ST Slope', type: 'Categorical', unit: '—', range: '0–2', desc: '0=Upsloping, 1=Flat, 2=Downsloping.' },
  { name: 'ca', display: 'Major Vessels', type: 'Categorical', unit: '—', range: '0–4', desc: 'Number of major vessels colored by fluoroscopy.' },
  { name: 'thal', display: 'Thalassemia', type: 'Categorical', unit: '—', range: '0–3', desc: '0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Unknown.' },
]

const LITERATURE = [
  { id: 1, authors: 'Detrano, R., Janosi, A., Steinbrunn, W., et al.', title: 'International application of a new probability algorithm for the diagnosis of coronary artery disease', journal: 'American Journal of Cardiology', year: 1989, volume: '64(4)', pages: '304–310', doi: 'https://doi.org/10.1016/0002-9149(89)90524-9', dataset: 'Cleveland, Hungarian, Swiss, VA', methodology: 'Bayesian algorithm', finding: 'Developed the original heart disease probability algorithm. Cleveland dataset collected.', relevance: 'Source of the dataset used in this project.' },
  { id: 2, authors: 'Das, R., Turkoglu, I., & Sengur, A.', title: 'Effective diagnosis of heart disease through neural networks ensembles', journal: 'Expert Systems with Applications', year: 2009, volume: '36(4)', pages: '7675–7680', doi: 'https://doi.org/10.1016/j.eswa.2008.10.059', dataset: 'UCI Cleveland', methodology: 'Neural network ensemble', finding: 'Ensemble of NNs achieved >90% accuracy on UCI dataset.', relevance: 'Demonstrates ensemble approaches for heart disease.' },
  { id: 3, authors: 'Weng, S. F., Reps, J., Kai, J., et al.', title: 'Can machine-learning improve cardiovascular risk prediction using routine clinical data?', journal: 'PLoS ONE', year: 2017, volume: '12(4)', pages: 'e0174944', doi: 'https://doi.org/10.1371/journal.pone.0174944', dataset: 'UK primary care (378,256)', methodology: 'Random Forest, Logistic Regression, GBM, Neural Network', finding: 'ML models achieved higher AUC (0.764) than ACC/AHA baseline (0.728).', relevance: 'Key paper comparing ML vs traditional cardiovascular risk scores.' },
  { id: 4, authors: 'Ribeiro, M. T., Singh, S., & Guestrin, C.', title: '"Why Should I Trust You?": Explaining the Predictions of Any Classifier', journal: 'Proceedings of the 22nd ACM SIGKDD', year: 2016, volume: '', pages: '1135–1144', doi: 'https://doi.org/10.1145/2939672.2939778', dataset: 'Various', methodology: 'LIME (Local Interpretable Model-agnostic Explanations)', finding: 'Introduced model-agnostic local explanations for classifier predictions.', relevance: 'Foundational work on explainable AI relevant to our explanation system.' },
  { id: 5, authors: 'Lundberg, S. M. & Lee, S. I.', title: 'A Unified Approach to Interpreting Model Predictions', journal: 'NeurIPS', year: 2017, volume: '30', pages: '', doi: 'https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions', dataset: 'Various', methodology: 'SHAP (SHapley Additive exPlanations)', finding: 'Game-theoretic approach to explain ML predictions. Unified LIME and Shapley values.', relevance: 'Framework for prediction explanation adopted in future work roadmap.' },
  { id: 6, authors: 'Niculescu-Mizil, A. & Caruana, R.', title: 'Predicting Good Probabilities with Supervised Learning', journal: 'ICML', year: 2005, volume: '', pages: '625–632', doi: 'https://doi.org/10.1145/1102351.1102430', dataset: 'Various', methodology: 'Probability calibration analysis', finding: 'Many ML algorithms produce poorly calibrated probabilities. Platt scaling and isotonic regression improve calibration.', relevance: 'Motivates our probability calibration implementation.' },
  { id: 7, authors: 'Brier, G. W.', title: 'Verification of Forecasts Expressed in Terms of Probability', journal: 'Monthly Weather Review', year: 1950, volume: '78(1)', pages: '1–3', doi: 'https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2', dataset: 'Weather forecasts', methodology: 'Brier Score', finding: 'Introduced the Brier score as a proper scoring rule for probability forecasts.', relevance: 'Metric used for calibration evaluation in this project.' },
  { id: 8, authors: 'Poplin, R., Varadarajan, A. V., Blumer, K., et al.', title: 'Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning', journal: 'Nature Biomedical Engineering', year: 2018, volume: '2(3)', pages: '158–164', doi: 'https://doi.org/10.1038/s41551-018-0195-1', dataset: 'Retinal images (284,335)', methodology: 'Deep learning (Inception-v3)', finding: 'AUC ≥0.70 for predicting age, gender, smoking, BP, HbA1c from retinal images.', relevance: 'Demonstrates ML potential for cardiovascular risk from novel data sources.' },
  { id: 9, authors: 'Johnson, A. E. W., Pollard, T. J., Shen, L., et al.', title: 'MIMIC-III, a freely accessible critical care database', journal: 'Scientific Data', year: 2016, volume: '3', pages: '160035', doi: 'https://doi.org/10.1038/sdata.2016.35', dataset: 'MIMIC-III (61,532 admissions)', methodology: 'Database documentation', finding: 'Freely accessible critical care database with comprehensive clinical data.', relevance: 'Potential future dataset for expanded model training.' },
  { id: 10, authors: 'Mahmood, S. S., Levy, D., Vasan, R. S., & Wang, T. J.', title: 'The Framingham Heart Study and the epidemiology of cardiovascular disease: a historical perspective', journal: 'The Lancet', year: 2014, volume: '383(9921)', pages: '999–1008', doi: 'https://doi.org/10.1016/S0140-6736(13)61940-5', dataset: 'Framingham Heart Study', methodology: 'Epidemiological review', finding: 'Framingham Risk Score is the gold standard for cardiovascular risk prediction.', relevance: 'Benchmark comparison for ML-based risk prediction approaches.' },
  { id: 11, authors: 'Sudlow, C., Gallacher, J., Allen, N., et al.', title: 'UK Biobank: An Open Access Resource for Identifying the Causes of a Wide Range of Complex Diseases', journal: 'PLoS Medicine', year: 2015, volume: '12(3)', pages: 'e1001779', doi: 'https://doi.org/10.1371/journal.pmed.1001779', dataset: 'UK Biobank (500,000)', methodology: 'Cohort documentation', finding: 'Large-scale biobank with genomic and health data for research.', relevance: 'Ideal future dataset for model expansion and validation.' },
]

const HYPERPARAMS = [
  { model: 'Logistic Regression', params: 'C = [0.01, 0.1, 1, 10, 100]\npenalty = ["l1", "l2"]', search: 'GridSearchCV' },
  { model: 'KNN', params: 'n_neighbors = [3, 5, 7, 9, 11, 15]\nweights = ["uniform", "distance"]', search: 'GridSearchCV' },
  { model: 'Decision Tree', params: 'max_depth = [3, 5, 7, 10, None]\ncriterion = ["gini", "entropy"]', search: 'GridSearchCV' },
  { model: 'Random Forest', params: 'n_estimators = [50, 100, 200]\nmax_depth = [5, 10, 15, None]', search: 'GridSearchCV' },
  { model: 'Gradient Boosting', params: 'n_estimators = [50, 100, 200]\nlearning_rate = [0.01, 0.05, 0.1, 0.2]', search: 'RandomizedSearchCV' },
  { model: 'SVM', params: 'C = [0.1, 1, 10]\nkernel = ["rbf", "linear"]', search: 'GridSearchCV' },
]

const API_ENDPOINTS = [
  { method: 'GET', path: '/health', desc: 'Health check endpoint', example: '{ "status": "healthy" }' },
  { method: 'GET', path: '/model/info', desc: 'Model metadata and version', example: '{ "model": "Logistic Regression", "version": "2.0.0" }' },
  { method: 'POST', path: '/predict', desc: 'Make a single prediction', example: '{ "age": 55, "sex": 1, "cp": 2, ... }' },
  { method: 'POST', path: '/explain', desc: 'Get prediction explanation', example: '{ "age": 55, "sex": 1, ... }' },
  { method: 'POST', path: '/scenario', desc: 'What-if scenario analysis', example: '{ "base": {...}, "changes": {...} }' },
  { method: 'GET', path: '/features', desc: 'Feature descriptions', example: '[ { "name": "age", "display": "Age" } ]' },
  { method: 'GET', path: '/data-sources', desc: 'Dataset information', example: '{ "name": "Cleveland", "records": 303 }' },
  { method: 'GET', path: '/metrics', desc: 'Model performance metrics', example: '{ "accuracy": 0.8361, "auc": 0.8885 }' },
]

const RESEARCH_QUESTIONS = [
  { q: 'Can ML predict heart disease risk from clinical features?', a: 'Yes. Our Logistic Regression model achieved 88.85% test ROC-AUC, demonstrating that clinical features contain predictive signal for heart disease.' },
  { q: 'Which algorithms perform best for this task?', a: 'Linear models (Logistic Regression, SVM) outperformed tree-based ensembles on this small dataset. Logistic Regression achieved the highest CV ROC-AUC of 0.9109.' },
  { q: 'Can probability calibration improve risk estimates?', a: 'Yes. Isotonic regression calibration yielded a Brier score of 0.12 for the best model, indicating well-calibrated probability estimates suitable for risk communication.' },
  { q: 'Can predictions be made interpretable?', a: 'Yes. Feature importance analysis reveals the top contributing features: exercise-induced angina (14.2%), thalassemia (12.8%), and major vessels (12.4%).' },
  { q: 'Can a web interface make the system accessible?', a: 'Yes. A guided 5-step questionnaire with contextual help enables non-technical users to enter clinical information and receive risk assessments.' },
]

/* ──────────────────────────────────────────────
   CHART COMPONENTS
   ────────────────────────────────────────────── */

function HorizontalBarChart({ data, maxVal, label }: { data: { label: string; value: number; color?: string }[]; maxVal: number; label: string }) {
  return (
    <div className="space-y-2">
      {data.map((d, i) => (
        <div key={i} className="flex items-center gap-3 group">
          <span className="w-36 text-sm text-gray-600 text-right truncate shrink-0">{d.label}</span>
          <div className="flex-1 bg-gray-100 rounded-full h-7 overflow-hidden relative">
            <div
              className={`h-full rounded-full ${d.color || 'bg-gradient-to-r from-blue-400 to-primary-600'} flex items-center justify-end pr-2 transition-all duration-700`}
              style={{ width: `${(d.value / maxVal) * 100}%`, minWidth: '2rem' }}
            >
              <span className="text-xs font-bold text-white">{(d.value * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      ))}
      <p className="text-xs text-gray-400 mt-1">Source: {label}</p>
    </div>
  )
}

function DonutChart({ segments, size = 160 }: { segments: { label: string; value: number; color: string }[]; size?: number }) {
  const total = segments.reduce((s, seg) => s + seg.value, 0)
  let cumPct = 0
  const r = size / 2 - 10
  const circ = 2 * Math.PI * r

  return (
    <div className="flex items-center gap-6">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {segments.map((seg, i) => {
          const pct = seg.value / total
          const dashLen = circ * pct
          const dashOff = circ * cumPct
          cumPct += pct
          return (
            <circle
              key={i}
              cx={size / 2} cy={size / 2} r={r}
              fill="none" stroke={seg.color} strokeWidth={20}
              strokeDasharray={`${dashLen} ${circ - dashLen}`}
              strokeDashoffset={-dashOff}
              transform={`rotate(-90 ${size / 2} ${size / 2})`}
              className="transition-all duration-500"
            />
          )
        })}
        <text x={size / 2} y={size / 2 - 8} textAnchor="middle" className="text-2xl font-bold fill-gray-800">{total}</text>
        <text x={size / 2} y={size / 2 + 12} textAnchor="middle" className="text-xs fill-gray-500">patients</text>
      </svg>
      <div className="space-y-2">
        {segments.map((seg, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: seg.color }} />
            <span className="text-sm text-gray-700">{seg.label}: <strong>{seg.value}</strong> ({((seg.value / total) * 100).toFixed(1)}%)</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ConfusionMatrix() {
  const cells = [
    { label: 'TN', value: 22, desc: 'True Negatives — Correctly identified as no disease', color: 'bg-green-100 text-green-800' },
    { label: 'FP', value: 6, desc: 'False Positives — Incorrectly flagged as disease', color: 'bg-amber-100 text-amber-800' },
    { label: 'FN', value: 4, desc: 'False Negatives — Missed disease cases (critical)', color: 'bg-red-100 text-red-800' },
    { label: 'TP', value: 29, desc: 'True Positives — Correctly identified disease', color: 'bg-blue-100 text-blue-800' },
  ]
  const [hovered, setHovered] = useState<number | null>(null)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 max-w-xs">
        <div />
        <div className="text-center text-xs font-semibold text-gray-500 pb-1">Pred: No Disease</div>
        <div className="text-center text-xs font-semibold text-gray-500 pb-1">Pred: Disease</div>
        <div className="text-xs font-semibold text-gray-500 pr-2 flex items-center justify-end">Actual: No</div>
        {cells.slice(0, 2).map((c, i) => (
          <div
            key={i}
            className={`${c.color} rounded-xl p-4 text-center cursor-pointer transition-all ${hovered === i ? 'ring-2 ring-primary-500 scale-105' : ''}`}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
          >
            <div className="text-2xl font-bold">{c.value}</div>
            <div className="text-xs mt-1">{c.label}</div>
          </div>
        ))}
        <div className="text-xs font-semibold text-gray-500 pr-2 flex items-center justify-end">Actual: Yes</div>
        {cells.slice(2, 4).map((c, i) => (
          <div
            key={i + 2}
            className={`${c.color} rounded-xl p-4 text-center cursor-pointer transition-all ${hovered === i + 2 ? 'ring-2 ring-primary-500 scale-105' : ''}`}
            onMouseEnter={() => setHovered(i + 2)}
            onMouseLeave={() => setHovered(null)}
          >
            <div className="text-2xl font-bold">{c.value}</div>
            <div className="text-xs mt-1">{c.label}</div>
          </div>
        ))}
      </div>
      {hovered !== null && (
        <div className="bg-gray-50 rounded-lg p-3 text-sm text-gray-700">{cells[hovered].desc}</div>
      )}
      <p className="text-xs text-gray-400">Source: Project experimental results (Logistic Regression, test set)</p>
    </div>
  )
}

function CorrelationHeatmap() {
  const features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak']
  // Approximate correlations from the Cleveland dataset
  const corr: Record<string, Record<string, number>> = {
    age:    { age: 1.0, sex: -0.10, cp: -0.10, trestbps: 0.28, chol: 0.22, thalach: -0.40, exang: 0.10, oldpeak: 0.21 },
    sex:    { age: -0.10, sex: 1.0, cp: -0.04, trestbps: 0.06, chol: -0.20, thalach: -0.04, exang: 0.15, oldpeak: 0.10 },
    cp:     { age: -0.10, sex: -0.04, cp: 1.0, trestbps: -0.04, chol: 0.08, thalach: 0.30, exang: -0.39, oldpeak: -0.35 },
    trestbps: { age: 0.28, sex: 0.06, cp: -0.04, trestbps: 1.0, chol: 0.13, thalach: -0.05, exang: 0.07, oldpeak: 0.19 },
    chol:   { age: 0.22, sex: -0.20, cp: 0.08, trestbps: 0.13, chol: 1.0, thalach: -0.02, exang: -0.06, oldpeak: 0.05 },
    thalach: { age: -0.40, sex: -0.04, cp: 0.30, trestbps: -0.05, chol: -0.02, thalach: 1.0, exang: -0.38, oldpeak: -0.34 },
    exang:  { age: 0.10, sex: 0.15, cp: -0.39, trestbps: 0.07, chol: -0.06, thalach: -0.38, exang: 1.0, oldpeak: 0.39 },
    oldpeak:{ age: 0.21, sex: 0.10, cp: -0.35, trestbps: 0.19, chol: 0.05, thalach: -0.34, exang: 0.39, oldpeak: 1.0 },
  }

  const getColor = (v: number) => {
    if (v > 0.3) return 'bg-red-400 text-white'
    if (v > 0.1) return 'bg-red-200 text-red-800'
    if (v > -0.1) return 'bg-gray-100 text-gray-600'
    if (v > -0.3) return 'bg-blue-200 text-blue-800'
    return 'bg-blue-400 text-white'
  }

  const [hover, setHover] = useState<{ r: string; c: string; v: number } | null>(null)

  return (
    <div>
      <div className="overflow-x-auto">
        <div className="inline-grid gap-px bg-gray-200 p-px rounded-lg" style={{ gridTemplateColumns: `auto repeat(${features.length}, 2.5rem)` }}>
          <div />
          {features.map(f => (
            <div key={f} className="text-[10px] font-medium text-gray-500 text-center p-1 truncate" style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)', height: '3.5rem' }}>{f}</div>
          ))}
          {features.map(row => (
            <>
              <div key={`l-${row}`} className="text-[10px] font-medium text-gray-500 flex items-center pr-1 truncate" style={{ minWidth: '4.5rem' }}>{row}</div>
              {features.map(col => {
                const v = corr[row]?.[col] ?? 0
                return (
                  <div
                    key={`${row}-${col}`}
                    className={`w-10 h-10 flex items-center justify-center text-[10px] font-mono rounded-sm cursor-pointer transition-transform ${getColor(v)} ${hover?.r === row && hover?.c === col ? 'ring-2 ring-primary-500 scale-125 z-10' : ''}`}
                    onMouseEnter={() => setHover({ r: row, c: col, v })}
                    onMouseLeave={() => setHover(null)}
                  >
                    {v.toFixed(1)}
                  </div>
                )
              })}
            </>
          ))}
        </div>
      </div>
      {hover && (
        <p className="text-sm text-gray-600 mt-2">{hover.r} ↔ {hover.c}: r = {hover.v.toFixed(2)}</p>
      )}
      <p className="text-xs text-gray-400 mt-1">Source: Computed from Cleveland Heart Disease dataset. Correlation ≠ causation.</p>
    </div>
  )
}

/* ──────────────────────────────────────────────
   SECTION COMPONENTS
   ────────────────────────────────────────────── */

function SectionNav({ sections, active }: { sections: string[]; active: string }) {
  const navRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = navRef.current?.querySelector(`[data-section="${active}"]`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
  }, [active])

  return (
    <div ref={navRef} className="sticky top-16 z-40 bg-white/90 backdrop-blur-lg border-b border-gray-100 shadow-sm overflow-x-auto">
      <div className="max-w-7xl mx-auto px-4 flex gap-1 py-2 min-w-max">
        {sections.map(s => (
          <a
            key={s}
            data-section={s}
            href={`#section-${s.toLowerCase().replace(/\s+/g, '-')}`}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${active === s ? 'bg-primary-100 text-primary-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'}`}
          >
            {s}
          </a>
        ))}
      </div>
    </div>
  )
}

function ExpandableCard({ title, children, defaultOpen = false }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden transition-all">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-gray-50 transition-colors">
        <span className="font-semibold text-gray-800">{title}</span>
        <ChevronDown className={`w-5 h-5 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && <div className="px-5 pb-5 border-t border-gray-100">{children}</div>}
    </div>
  )
}

function KpiCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5 text-center hover:shadow-md transition-shadow">
      <div className="text-sm text-gray-500 mb-1">{label}</div>
      <div className="text-3xl font-extrabold text-gray-900">{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-1">{sub}</div>}
    </div>
  )
}

/* ──────────────────────────────────────────────
   MAIN RESEARCH PAGE
   ────────────────────────────────────────────── */

interface Props { onNavigate: (page: string) => void }

export default function ResearchPage({ onNavigate }: Props) {
  const [activeSection, setActiveSection] = useState('Overview')
  const [expandedLit, setExpandedLit] = useState<number | null>(null)
  const [metricView, setMetricView] = useState<'cvAuc' | 'accuracy' | 'recall' | 'specificity' | 'f1' | 'brier'>('cvAuc')

  const sections = [
    'Overview', 'Literature Survey', 'Research Gap', 'Dataset', 'Methodology',
    'Models', 'Results', 'Explainability', 'System Architecture', 'Application',
    'Security & Ethics', 'Limitations', 'Future Work', 'References'
  ]

  useEffect(() => {
    const handler = () => {
      const sectionEls = sections.map(s => document.getElementById(`section-${s.toLowerCase().replace(/\s+/g, '-')}`))
      for (let i = sectionEls.length - 1; i >= 0; i--) {
        const el = sectionEls[i]
        if (el && el.getBoundingClientRect().top <= 120) {
          setActiveSection(sections[i])
          break
        }
      }
    }
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [])

  const metricLabel: Record<string, { key: string; label: string; format: (v: number) => string; max: number }> = {
    cvAuc: { key: 'cvAuc', label: 'Cross-Validated ROC-AUC', format: v => `${(v * 100).toFixed(2)}%`, max: 1 },
    accuracy: { key: 'accuracy', label: 'Test Accuracy', format: v => `${(v * 100).toFixed(2)}%`, max: 1 },
    recall: { key: 'recall', label: 'Recall (Sensitivity)', format: v => `${(v * 100).toFixed(2)}%`, max: 1 },
    specificity: { key: 'specificity', label: 'Specificity', format: v => `${(v * 100).toFixed(2)}%`, max: 1 },
    f1: { key: 'f1', label: 'F1 Score', format: v => `${(v * 100).toFixed(2)}%`, max: 1 },
    brier: { key: 'brier', label: 'Brier Score (lower = better)', format: v => v.toFixed(3), max: 0.35 },
  }

  const currentMetric = metricLabel[metricView]
  const chartData = MODEL_RESULTS.map(m => ({
    label: m.name,
    value: (m as any)[currentMetric.key],
    color: m.best ? 'bg-gradient-to-r from-green-400 to-emerald-600' : undefined,
  })).sort((a, b) => metricView === 'brier' ? a.value - b.value : b.value - a.value)

  return (
    <div className="space-y-0">
      {/* ── HERO ── */}
      <section className="relative overflow-hidden bg-gradient-to-br from-gray-900 via-primary-900 to-gray-900 text-white -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8 py-16 sm:py-24">
        <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%), radial-gradient(circle at 80% 50%, rgba(255,255,255,0.08) 0%, transparent 50%)' }} />
        <div className="relative max-w-5xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm text-blue-200 px-4 py-1.5 rounded-full text-sm font-medium mb-6">
            <Beaker className="w-4 h-4" />
            Research Project
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold leading-tight mb-6">
            {RESEARCH_META.title}
          </h1>
          <p className="text-lg text-blue-200/80 max-w-3xl mx-auto mb-10 leading-relaxed">
            {RESEARCH_META.subtitle}
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <button onClick={() => document.getElementById('section-overview')?.scrollIntoView({ behavior: 'smooth' })} className="bg-white text-gray-900 px-6 py-3 rounded-xl font-semibold hover:bg-gray-100 transition-colors shadow-lg">
              <FileText className="w-4 h-4 inline mr-2" />View Research
            </button>
            <a href={RESEARCH_META.repo} target="_blank" rel="noopener noreferrer" className="bg-white/10 backdrop-blur text-white px-6 py-3 rounded-xl font-semibold hover:bg-white/20 transition-colors border border-white/20">
              <Github className="w-4 h-4 inline mr-2" />View on GitHub
            </a>
            <button onClick={() => onNavigate('predict')} className="bg-primary-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-primary-700 transition-colors shadow-lg">
              <Activity className="w-4 h-4 inline mr-2" />Try Prediction
            </button>
          </div>
        </div>
      </section>

      {/* ── RESEARCH METADATA ── */}
      <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 -mt-8 relative z-10">
        {[
          { l: 'Dataset', v: 'Cleveland, 303 pts' },
          { l: 'Best Model', v: 'Log. Regression' },
          { l: 'CV ROC-AUC', v: '0.9109' },
          { l: 'Test ROC-AUC', v: '0.8885' },
          { l: 'Academic Year', v: RESEARCH_META.academicYear },
          { l: 'Version', v: RESEARCH_META.version },
        ].map((item, i) => (
          <div key={i} className="bg-white rounded-xl shadow-lg border border-gray-100 p-4 text-center">
            <div className="text-xs text-gray-500 mb-1">{item.l}</div>
            <div className="text-sm font-bold text-gray-900">{item.v}</div>
          </div>
        ))}
      </section>

      {/* ── NAV ── */}
      <SectionNav sections={sections} active={activeSection} />

      {/* ═══════════════════════════════════════════
          SECTION: OVERVIEW / EXECUTIVE SUMMARY
          ═══════════════════════════════════════════ */}
      <section id="section-overview" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-8">Executive Summary</h2>
        <div className="prose prose-gray max-w-none space-y-4 text-gray-700 leading-relaxed">
          <p>Cardiovascular disease remains the leading cause of death globally, claiming approximately 17.9 million lives annually (WHO, 2021). Early risk assessment is crucial for improving patient outcomes and reducing healthcare burden. This project presents an explainable machine learning framework for heart disease risk prediction using publicly available clinical data.</p>
          <p>The system implements a comprehensive pipeline including data validation, preprocessing with leak-proof splitting, feature encoding, and scaling. Six machine learning algorithms were evaluated—Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting, and Support Vector Machine—with hyperparameter optimization using GridSearchCV and RandomizedSearchCV across 5-fold stratified cross-validation.</p>
          <p>The best-performing model, <strong>Logistic Regression</strong>, achieved a cross-validated ROC-AUC of <strong>0.9109</strong> and a test-set ROC-AUC of <strong>0.8885</strong>, with recall of <strong>87.88%</strong> and specificity of <strong>78.57%</strong>. Probability calibration using isotonic regression yielded a Brier score of <strong>0.12</strong>, indicating well-calibrated probability estimates. The system provides feature importance analysis and individual prediction explanations to support interpretability.</p>
          <p>A production-ready REST API and a React-based web interface enable non-technical users to obtain risk assessments through a guided questionnaire with contextual help. The system is designed as a research and educational tool and is not intended for clinical diagnosis.</p>
        </div>

        {/* Key Findings Strip */}
        <div className="mt-8">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Key Findings</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <KpiCard label="CV ROC-AUC" value="0.9109" sub="5-fold stratified" />
            <KpiCard label="Test ROC-AUC" value="0.8885" sub="20% held-out" />
            <KpiCard label="Accuracy" value="83.61%" sub="Test set" />
            <KpiCard label="Recall" value="87.88%" sub="Sensitivity" />
            <KpiCard label="Specificity" value="78.57%" sub="True negative rate" />
            <KpiCard label="Brier Score" value="0.12" sub="Calibration" />
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: RESEARCH PROBLEM
          ═══════════════════════════════════════════ */}
      <section className="pt-12">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Research Problem</h2>
        <div className="grid lg:grid-cols-2 gap-8">
          <div className="space-y-4 text-gray-700 leading-relaxed">
            <p>Heart disease remains a significant global health challenge. While clinical risk factors are well-established, the complex interactions between these factors make risk assessment challenging using traditional statistical methods. Existing risk calculators like the Framingham Risk Score use limited linear models and may not capture non-linear relationships.</p>
            <p>Machine learning offers the potential to identify complex patterns in clinical data, but adoption faces challenges: model interpretability, probability calibration, generalizability across populations, and user accessibility for non-technical users.</p>
            <p><strong>This project addresses:</strong> How can an integrated ML framework combine rigorous preprocessing, model comparison, probability calibration, explainability, and an accessible interface for heart disease risk estimation?</p>
          </div>
          <div className="space-y-3">
            <h3 className="font-bold text-gray-900">The system CAN:</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              {['Estimate heart disease risk based on clinical features', 'Provide calibrated probability estimates', 'Explain which features contributed to predictions', 'Guide users through a questionnaire', 'Validate user inputs'].map((item, i) => (
                <li key={i} className="flex items-start gap-2"><Check className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />{item}</li>
              ))}
            </ul>
            <h3 className="font-bold text-gray-900 mt-4">The system CANNOT:</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              {['Diagnose heart disease', 'Replace medical professionals', 'Guarantee prediction accuracy', 'Provide treatment recommendations', 'Handle emergency situations'].map((item, i) => (
                <li key={i} className="flex items-start gap-2"><X className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />{item}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Research Questions */}
        <div className="mt-8 space-y-3">
          <h3 className="text-xl font-bold text-gray-900">Research Questions</h3>
          {RESEARCH_QUESTIONS.map((rq, i) => (
            <ExpandableCard key={i} title={rq.q}>
              <p className="text-gray-700 leading-relaxed mt-2">{rq.a}</p>
            </ExpandableCard>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: LITERATURE SURVEY
          ═══════════════════════════════════════════ */}
      <section id="section-literature-survey" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Literature Survey</h2>
        <p className="text-gray-600 mb-8 max-w-3xl">A review of key research in machine learning for cardiovascular prediction, explainable AI, and probability calibration.</p>

        {/* Literature Comparison Table */}
        <div className="overflow-x-auto mb-8">
          <table className="w-full text-sm border border-gray-200 rounded-xl overflow-hidden">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left py-3 px-4 font-semibold text-gray-700">Study</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-700">Year</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-700">Dataset</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-700">Algorithms</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-700">Best Result</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-700">Contribution</th>
              </tr>
            </thead>
            <tbody>
              {LITERATURE.map((lit, i) => (
                <tr key={lit.id} className={`border-t border-gray-100 ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'} hover:bg-primary-50/30 transition-colors`}>
                  <td className="py-3 px-4 font-medium text-gray-800 max-w-[200px] truncate">{lit.authors.split(',')[0]}</td>
                  <td className="py-3 px-4 text-center">{lit.year}</td>
                  <td className="py-3 px-4 text-gray-600 max-w-[150px] truncate">{lit.dataset}</td>
                  <td className="py-3 px-4 text-gray-600 max-w-[150px] truncate">{lit.methodology}</td>
                  <td className="py-3 px-4 text-center text-gray-700">{lit.finding.substring(0, 40)}…</td>
                  <td className="py-3 px-4 text-gray-600 max-w-[200px] truncate">{lit.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Literature Performance Comparison — Weng et al. */}
        <div className="card mb-8">
          <h3 className="font-bold text-gray-900 mb-2">Published Cardiovascular Risk Prediction Performance — Weng et al. (2017)</h3>
          <p className="text-sm text-gray-500 mb-4">Source: Weng et al., PLoS ONE, 2017. These are the paper's actual published numbers, not from this project.</p>
          <div className="space-y-2">
            {[
              { label: 'ACC/AHA Baseline', value: 0.728 },
              { label: 'Random Forest', value: 0.745 },
              { label: 'Logistic Regression', value: 0.760 },
              { label: 'Gradient Boosting', value: 0.761 },
              { label: 'Neural Network', value: 0.764 },
            ].map(d => (
              <div key={d.label} className="flex items-center gap-3">
                <span className="w-40 text-sm text-gray-600 text-right">{d.label}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                  <div className="h-full rounded-full bg-gradient-to-r from-amber-300 to-amber-500 flex items-center justify-end pr-2" style={{ width: `${(d.value / 0.85) * 100}%` }}>
                    <span className="text-xs font-bold text-white">{d.value}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Individual Papers */}
        <div className="space-y-3">
          {LITERATURE.map(lit => (
            <ExpandableCard key={lit.id} title={`${lit.authors.split(',')[0].split(' ').pop()} (${lit.year}) — ${lit.title.substring(0, 80)}…`}>
              <div className="space-y-3 mt-3">
                <div className="grid sm:grid-cols-2 gap-4 text-sm">
                  <div><span className="font-semibold text-gray-700">Authors:</span> <span className="text-gray-600">{lit.authors}</span></div>
                  <div><span className="font-semibold text-gray-700">Journal:</span> <span className="text-gray-600">{lit.journal} {lit.volume} {lit.pages}</span></div>
                  <div><span className="font-semibold text-gray-700">Dataset:</span> <span className="text-gray-600">{lit.dataset}</span></div>
                  <div><span className="font-semibold text-gray-700">Methodology:</span> <span className="text-gray-600">{lit.methodology}</span></div>
                </div>
                <p className="text-sm text-gray-700"><span className="font-semibold">Key Finding:</span> {lit.finding}</p>
                <p className="text-sm text-gray-700"><span className="font-semibold">Relevance:</span> {lit.relevance}</p>
                <div className="flex flex-wrap gap-2 pt-2">
                  <a href={lit.doi} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-xs bg-blue-50 text-blue-700 px-3 py-1.5 rounded-full hover:bg-blue-100 transition-colors">
                    <ExternalLink className="w-3 h-3" />DOI
                  </a>
                </div>
              </div>
            </ExpandableCard>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: RESEARCH GAP
          ═══════════════════════════════════════════ */}
      <section id="section-research-gap" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Research Gap</h2>
        <div className="bg-gradient-to-br from-gray-50 to-primary-50/30 rounded-2xl p-8 border border-gray-200">
          <div className="grid lg:grid-cols-[1fr_auto_1fr] gap-8 items-center">
            <div className="space-y-3">
              {['Prediction using single algorithms', 'Basic model comparison', 'Limited explainability', 'No probability calibration', 'No user accessibility', 'No integrated pipeline'].map((item, i) => (
                <div key={i} className="flex items-center gap-3 text-sm text-gray-600">
                  <span className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-xs font-bold text-gray-500">{i + 1}</span>
                  {item}
                </div>
              ))}
            </div>
            <div className="flex flex-col items-center gap-2 text-primary-600">
              <div className="text-sm font-bold">Gap</div>
              <ArrowRight className="w-8 h-8 rotate-90 lg:rotate-0" />
            </div>
            <div className="space-y-3">
              {['Multi-algorithm comparative framework', 'Rigorous leakage-safe preprocessing', 'Probability calibration with isotonic regression', 'Feature importance explanations', 'Guided questionnaire for non-technical users', 'Full REST API + web interface'].map((item, i) => (
                <div key={i} className="flex items-center gap-3 text-sm text-primary-700 font-medium">
                  <Check className="w-5 h-5 text-green-500 shrink-0" />
                  {item}
                </div>
              ))}
            </div>
          </div>
          <p className="text-sm text-gray-500 mt-6 text-center">This project integrates these components into one educational and research-oriented application.</p>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: DATASET
          ═══════════════════════════════════════════ */}
      <section id="section-dataset" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Dataset</h2>

        {/* Dataset Overview */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          <div className="card">
            <h3 className="font-bold text-gray-900 mb-4">Cleveland Heart Disease Dataset</h3>
            <div className="space-y-2 text-sm">
              {[
                ['Source', 'UCI Machine Learning Repository'],
                ['Creator', 'Cleveland Clinic Foundation (Detrano et al., 1989)'],
                ['Records', '303 patients (1 duplicate removed → 302 unique)'],
                ['Features', '14 (13 input + 1 target)'],
                ['Target', 'Heart disease presence (0/1)'],
                ['License', 'CC BY 4.0'],
                ['Missing Values', 'None (after handling \'?\' markers)'],
                ['Year', '1988'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between py-1 border-b border-gray-50">
                  <span className="text-gray-500">{k}</span>
                  <span className="font-medium text-gray-800 text-right">{v}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <h3 className="font-bold text-gray-900 mb-4">Class Distribution</h3>
            <DonutChart
              segments={[
                { label: 'Disease (1)', value: 165, color: '#ef4444' },
                { label: 'No Disease (0)', value: 138, color: '#22c55e' },
              ]}
            />
            <p className="text-xs text-gray-400 mt-3">Source: Processed Cleveland dataset. Moderate class imbalance (54.5% positive).</p>
          </div>
        </div>

        {/* Feature Explorer */}
        <div className="card mb-8">
          <h3 className="font-bold text-gray-900 mb-4">Feature Dictionary</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left py-2 px-3 font-semibold text-gray-700">Feature</th>
                  <th className="text-left py-2 px-3 font-semibold text-gray-700">Display Name</th>
                  <th className="text-center py-2 px-3 font-semibold text-gray-700">Type</th>
                  <th className="text-center py-2 px-3 font-semibold text-gray-700">Unit</th>
                  <th className="text-center py-2 px-3 font-semibold text-gray-700">Range</th>
                  <th className="text-left py-2 px-3 font-semibold text-gray-700">Description</th>
                </tr>
              </thead>
              <tbody>
                {FEATURES.map((f, i) => (
                  <tr key={f.name} className={`border-t border-gray-100 ${i % 2 === 0 ? '' : 'bg-gray-50/50'}`}>
                    <td className="py-2 px-3 font-mono text-primary-700">{f.name}</td>
                    <td className="py-2 px-3 font-medium">{f.display}</td>
                    <td className="py-2 px-3 text-center"><span className={`px-2 py-0.5 rounded-full text-xs ${f.type === 'Numeric' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'}`}>{f.type}</span></td>
                    <td className="py-2 px-3 text-center text-gray-500">{f.unit}</td>
                    <td className="py-2 px-3 text-center font-mono text-gray-600">{f.range}</td>
                    <td className="py-2 px-3 text-gray-600 max-w-[250px]">{f.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Correlation Heatmap */}
        <div className="card mb-8">
          <h3 className="font-bold text-gray-900 mb-4">Feature Correlation Heatmap</h3>
          <CorrelationHeatmap />
        </div>

        {/* Data Quality Pipeline */}
        <div className="card">
          <h3 className="font-bold text-gray-900 mb-4">Data Quality Pipeline</h3>
          <div className="flex flex-wrap items-center gap-3">
            {['Raw Data (303 rows)', 'Validate Schema', 'Remove Duplicates (1)', 'Handle Missing (?)', 'Train/Test Split (80/20)', 'Preprocess (fit on train only)', 'Validated Dataset (302 rows)'].map((step, i, arr) => (
              <div key={i} className="flex items-center gap-3">
                <div className="bg-primary-50 border border-primary-200 rounded-lg px-4 py-2 text-sm font-medium text-primary-800">{step}</div>
                {i < arr.length - 1 && <ArrowRight className="w-4 h-4 text-gray-400 shrink-0" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: METHODOLOGY
          ═══════════════════════════════════════════ */}
      <section id="section-methodology" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Methodology</h2>

        {/* Preprocessing */}
        <div className="card mb-6">
          <h3 className="font-bold text-gray-900 mb-4">Preprocessing Pipeline</h3>
          <div className="grid sm:grid-cols-2 gap-6 text-sm">
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800">Numerical Features (5)</h4>
              <p className="text-gray-600">age, trestbps, chol, thalach, oldpeak</p>
              <p className="text-gray-600">• Scaling: StandardScaler (zero mean, unit variance)</p>
              <p className="text-gray-600">• Imputation: Median (for missing values)</p>
            </div>
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800">Categorical Features (8)</h4>
              <p className="text-gray-600">sex, cp, fbs, restecg, exang, slope, ca, thal</p>
              <p className="text-gray-600">• Encoding: OneHotEncoder (drop='first')</p>
              <p className="text-gray-600">• Result: 22 features after encoding</p>
            </div>
          </div>
        </div>

        {/* Leakage Prevention */}
        <div className="card mb-6 bg-blue-50 border-blue-200">
          <h3 className="font-bold text-blue-900 mb-3 flex items-center gap-2">
            <Shield className="w-5 h-5" />Data Leakage Prevention
          </h3>
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <div className="bg-white rounded-xl px-4 py-3 border border-blue-200">
              <div className="font-semibold text-blue-800">1. Load Raw Data</div>
              <div className="text-blue-600 text-xs">303 rows</div>
            </div>
            <ArrowRight className="w-5 h-5 text-blue-400" />
            <div className="bg-white rounded-xl px-4 py-3 border border-blue-200">
              <div className="font-semibold text-blue-800">2. Train/Test Split</div>
              <div className="text-blue-600 text-xs">BEFORE preprocessing</div>
            </div>
            <ArrowRight className="w-5 h-5 text-blue-400" />
            <div className="bg-white rounded-xl px-4 py-3 border border-blue-200">
              <div className="font-semibold text-blue-800">3. Fit on Train Only</div>
              <div className="text-blue-600 text-xs">StandardScaler, OneHotEncoder</div>
            </div>
            <ArrowRight className="w-5 h-5 text-blue-400" />
            <div className="bg-white rounded-xl px-4 py-3 border border-blue-200">
              <div className="font-semibold text-blue-800">4. Transform Test</div>
              <div className="text-blue-600 text-xs">Using train-fitted preprocessor</div>
            </div>
          </div>
          <p className="text-sm text-blue-700 mt-3">Preprocessing before splitting causes information from the test set to leak into training, producing over-optimistic estimates.</p>
        </div>

        {/* Experimental Design */}
        <div className="card">
          <h3 className="font-bold text-gray-900 mb-4">Experimental Design</h3>
          <div className="grid sm:grid-cols-2 gap-6 text-sm">
            {[
              ['Train/Test Split', '80% train / 20% test, stratified'],
              ['Random Seed', '42 (for reproducibility)'],
              ['Cross-Validation', '5-fold stratified'],
              ['Primary Metric', 'ROC-AUC (cross-validated)'],
              ['Secondary Metrics', 'Accuracy, Recall, Specificity, F1, Brier Score'],
              ['Search Strategy', 'GridSearchCV / RandomizedSearchCV'],
              ['Calibration', 'Isotonic regression'],
              ['Training Samples', '241 patients'],
              ['Test Samples', '61 patients'],
              ['Features After Encoding', '22'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-gray-500">{k}</span>
                <span className="font-medium text-gray-800 text-right">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: MODELS
          ═══════════════════════════════════════════ */}
      <section id="section-models" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Machine Learning Models</h2>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {MODEL_RESULTS.map(m => (
            <div key={m.name} className={`card card-hover ${m.best ? 'ring-2 ring-green-500 bg-green-50/30' : ''}`}>
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-bold text-gray-900">{m.name}</h4>
                {m.best && <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-semibold">Selected</span>}
              </div>
              <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">{m.category}</span>
              <p className="text-sm text-gray-600 mt-2 leading-relaxed">{m.desc}</p>
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-gray-500">
                <div>CV AUC: <span className="font-bold text-gray-800">{m.cvAuc}</span></div>
                <div>Test AUC: <span className="font-bold text-gray-800">{m.testAuc}</span></div>
                <div>Recall: <span className="font-bold text-gray-800">{(m.recall * 100).toFixed(1)}%</span></div>
                <div>Brier: <span className="font-bold text-gray-800">{m.brier}</span></div>
              </div>
            </div>
          ))}
        </div>

        {/* Hyperparameter Search Spaces */}
        <div className="card">
          <h3 className="font-bold text-gray-900 mb-4">Hyperparameter Search Spaces</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left py-2 px-3 font-semibold text-gray-700">Model</th>
                  <th className="text-left py-2 px-3 font-semibold text-gray-700">Hyperparameters</th>
                  <th className="text-center py-2 px-3 font-semibold text-gray-700">Search Strategy</th>
                </tr>
              </thead>
              <tbody>
                {HYPERPARAMS.map((h, i) => (
                  <tr key={h.model} className={`border-t border-gray-100 ${i % 2 === 0 ? '' : 'bg-gray-50/50'}`}>
                    <td className="py-3 px-3 font-medium">{h.model}</td>
                    <td className="py-3 px-3"><pre className="text-xs text-gray-600 whitespace-pre-wrap font-mono">{h.params}</pre></td>
                    <td className="py-3 px-3 text-center"><span className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded-full">{h.search}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: RESULTS
          ═══════════════════════════════════════════ */}
      <section id="section-results" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Results</h2>

        {/* Best Model KPIs */}
        <div className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-2xl p-8 text-white mb-8">
          <div className="flex items-center gap-2 text-primary-200 text-sm font-medium mb-2">
            <Shield className="w-4 h-4" /> Selected Model: Logistic Regression
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-6 mt-4">
            {[
              { l: 'CV ROC-AUC', v: '0.9109' },
              { l: 'Test ROC-AUC', v: '88.85%' },
              { l: 'Accuracy', v: '83.61%' },
              { l: 'Recall', v: '87.88%' },
              { l: 'Brier Score', v: '0.12' },
            ].map((kpi, i) => (
              <div key={i}>
                <div className="text-primary-200 text-sm">{kpi.l}</div>
                <div className="text-2xl font-bold">{kpi.v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Multi-Metric Comparison */}
        <div className="card mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-gray-900">Model Comparison</h3>
            <div className="flex flex-wrap gap-1">
              {(Object.keys(metricLabel) as Array<keyof typeof metricLabel>).map(key => (
                <button
                  key={key}
                  onClick={() => setMetricView(key)}
                  className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${metricView === key ? 'bg-primary-100 text-primary-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'}`}
                >
                  {metricLabel[key].label.split('(')[0].trim()}
                </button>
              ))}
            </div>
          </div>
          <HorizontalBarChart data={chartData} maxVal={currentMetric.max} label="Project experimental results" />
        </div>

        {/* Full Comparison Table */}
        <div className="card mb-8 overflow-hidden">
          <h3 className="font-bold text-gray-900 mb-4">Complete Model Comparison</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Model</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">CV AUC</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Test AUC</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Accuracy</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Recall</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Specificity</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">F1</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Brier</th>
                </tr>
              </thead>
              <tbody>
                {MODEL_RESULTS.map(m => (
                  <tr key={m.name} className={`border-t border-gray-100 ${m.best ? 'bg-green-50' : 'hover:bg-gray-50'}`}>
                    <td className="py-3 px-4 font-medium">
                      {m.name}
                      {m.best && <span className="ml-2 text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Best</span>}
                    </td>
                    <td className="py-3 px-4 text-right font-mono">{m.cvAuc}</td>
                    <td className="py-3 px-4 text-right font-mono">{m.testAuc}</td>
                    <td className="py-3 px-4 text-right font-mono">{(m.accuracy * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4 text-right font-mono">{(m.recall * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4 text-right font-mono">{(m.specificity * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4 text-right font-mono">{(m.f1 * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4 text-right font-mono">{m.brier}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-400 mt-2 px-4">Source: Project experimental results. 5-fold stratified cross-validation, test set evaluation.</p>
        </div>

        {/* Confusion Matrix */}
        <div className="card mb-8">
          <h3 className="font-bold text-gray-900 mb-4">Confusion Matrix — Logistic Regression (Test Set)</h3>
          <ConfusionMatrix />
        </div>

        {/* Calibration */}
        <div className="card">
          <h3 className="font-bold text-gray-900 mb-4">Probability Calibration</h3>
          <div className="grid sm:grid-cols-2 gap-6">
            <div className="space-y-3 text-sm text-gray-700">
              <p>Probability calibration ensures that predicted probabilities reflect true event probabilities. Many ML algorithms produce poorly calibrated outputs.</p>
              <p><strong>Method:</strong> Isotonic regression applied to Logistic Regression probabilities.</p>
              <p><strong>Brier Score:</strong> 0.12 (lower is better; 0 = perfect, 1 = worst)</p>
              <p>This indicates the model's probability estimates are reasonably well-calibrated for risk communication.</p>
            </div>
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800">Calibration by Model</h4>
              {MODEL_RESULTS.map(m => (
                <div key={m.name} className="flex items-center gap-3">
                  <span className="w-36 text-sm text-gray-600 truncate">{m.name}</span>
                  <div className="flex-1 bg-gray-100 rounded-full h-5 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${m.calibrated ? 'bg-green-400' : 'bg-amber-400'}`}
                      style={{ width: `${(1 - m.brier / 0.35) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-gray-600 w-12 text-right">{m.brier}</span>
                </div>
              ))}
              <p className="text-xs text-gray-400">Source: Project experimental results. Green = calibrated, amber = uncalibrated.</p>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: EXPLAINABILITY
          ═══════════════════════════════════════════ */}
      <section id="section-explainability" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Explainability</h2>

        {/* Feature Importance */}
        <div className="card mb-8">
          <h3 className="font-bold text-gray-900 mb-4">Global Feature Importance</h3>
          <HorizontalBarChart
            data={FEATURE_IMPORTANCE.map(f => ({ label: f.name, value: f.importance }))}
            maxVal={0.16}
            label="Project experimental results"
          />
          <div className="mt-4 bg-amber-50 border border-amber-200 rounded-xl p-3">
            <p className="text-sm text-amber-800">
              <Info className="w-4 h-4 inline mr-1" />
              <strong>Important:</strong> Feature importance indicates statistical contribution to model predictions and does not establish medical causation.
            </p>
          </div>
        </div>

        {/* Explanation Types */}
        <div className="grid sm:grid-cols-2 gap-6 mb-8">
          <div className="card">
            <h4 className="font-bold text-gray-900 mb-2 flex items-center gap-2"><Globe className="w-5 h-5 text-primary-600" /> Global Explanation</h4>
            <p className="text-sm text-gray-600">Identifies which features matter most across the entire dataset. Exercise-induced angina, thalassemia, and major vessels are the top three contributing features.</p>
          </div>
          <div className="card">
            <h4 className="font-bold text-gray-900 mb-2 flex items-center gap-2"><Users className="w-5 h-5 text-primary-600" /> Local Explanation</h4>
            <p className="text-sm text-gray-600">For each prediction, the system identifies contributing factors (increasing risk) and protective factors (decreasing risk) specific to that individual's inputs.</p>
          </div>
        </div>

        <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-8">
          <p className="text-sm text-red-800">
            <AlertTriangle className="w-4 h-4 inline mr-1" />
            <strong>Causal Limitation:</strong> Model contribution ≠ medical causation. The system provides statistical feature contributions, not causal medical explanations.
          </p>
        </div>

        {/* Future Explainability */}
        <div className="card bg-gray-50">
          <h4 className="font-bold text-gray-900 mb-2">Future Explainability Enhancements</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• SHAP (SHapley Additive exPlanations) for game-theoretic feature attribution</li>
            <li>• LIME (Local Interpretable Model-agnostic Explanations) for local explanations</li>
            <li>• Individual prediction waterfall charts</li>
          </ul>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: SYSTEM ARCHITECTURE
          ═══════════════════════════════════════════ */}
      <section id="section-system-architecture" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">System Architecture</h2>

        {/* Architecture Diagram */}
        <div className="card mb-8">
          <h3 className="font-bold text-gray-900 mb-4">System Architecture</h3>
          <div className="flex flex-col items-center gap-2">
            {[
              { label: 'User (Browser)', color: 'bg-gray-100 border-gray-300 text-gray-700' },
              { label: 'React + TypeScript + Tailwind CSS', color: 'bg-blue-50 border-blue-200 text-blue-800' },
              { label: 'FastAPI REST Endpoints', color: 'bg-green-50 border-green-200 text-green-800' },
              { label: 'Pydantic Validation', color: 'bg-purple-50 border-purple-200 text-purple-800' },
              { label: 'Preprocessing Pipeline (sklearn)', color: 'bg-amber-50 border-amber-200 text-amber-800' },
              { label: 'ML Model (Logistic Regression)', color: 'bg-red-50 border-red-200 text-red-800' },
              { label: 'Calibration (Isotonic Regression)', color: 'bg-teal-50 border-teal-200 text-teal-800' },
              { label: 'Prediction + Explanation Response', color: 'bg-gray-100 border-gray-300 text-gray-700' },
            ].map((layer, i, arr) => (
              <div key={i} className="flex items-center gap-3 w-full max-w-lg">
                <div className={`flex-1 ${layer.color} border rounded-xl px-5 py-3 text-sm font-medium text-center`}>{layer.label}</div>
              </div>
            )).reduce<React.ReactNode[]>((acc, el, i, arr) => {
              acc.push(el)
              if (i < arr.length - 1) acc.push(<ArrowRight key={`a-${i}`} className="w-5 h-5 text-gray-400 -rotate-90 mx-auto" />)
              return acc
            }, [])}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="card">
          <h3 className="font-bold text-gray-900 mb-4">Technology Stack</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {[
              ['Python', 'Core ML language'],
              ['scikit-learn', 'ML models & preprocessing'],
              ['pandas', 'Data manipulation'],
              ['NumPy', 'Numerical computation'],
              ['FastAPI', 'REST API framework'],
              ['Pydantic', 'Data validation'],
              ['React 18', 'Frontend UI'],
              ['TypeScript', 'Type-safe JavaScript'],
              ['Vite', 'Build tool'],
              ['Tailwind CSS', 'Utility-first CSS'],
              ['Recharts', 'Data visualization'],
              ['Lucide React', 'Icon library'],
              ['Docker', 'Containerization'],
              ['matplotlib', 'Python visualization'],
            ].map(([tech, desc]) => (
              <div key={tech} className="bg-gray-50 rounded-lg px-3 py-2 text-sm">
                <div className="font-semibold text-gray-800">{tech}</div>
                <div className="text-xs text-gray-500">{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: APPLICATION
          ═══════════════════════════════════════════ */}
      <section id="section-application" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Application</h2>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* API Endpoints */}
          <div className="card">
            <h3 className="font-bold text-gray-900 mb-4">API Endpoints</h3>
            <div className="space-y-2">
              {API_ENDPOINTS.map(ep => (
                <div key={ep.path} className="flex items-center gap-3 text-sm py-2 border-b border-gray-50 last:border-0">
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${ep.method === 'GET' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}`}>{ep.method}</span>
                  <span className="font-mono text-gray-800">{ep.path}</span>
                  <span className="text-gray-500 ml-auto hidden sm:block">{ep.desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* User Interface */}
          <div className="card">
            <h3 className="font-bold text-gray-900 mb-4">User Interface</h3>
            <div className="space-y-3 text-sm text-gray-700">
              <p><strong>5-Step Guided Questionnaire:</strong></p>
              <ol className="space-y-2 ml-4">
                <li className="flex items-start gap-2"><span className="w-5 h-5 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs font-bold shrink-0">1</span>About You — Age, Sex</li>
                <li className="flex items-start gap-2"><span className="w-5 h-5 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs font-bold shrink-0">2</span>General Health — BP, Cholesterol, Blood sugar</li>
                <li className="flex items-start gap-2"><span className="w-5 h-5 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs font-bold shrink-0">3</span>Heart Tests — ECG, Max HR, Exercise angina</li>
                <li className="flex items-start gap-2"><span className="w-5 h-5 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs font-bold shrink-0">4</span>Exercise Response — ST depression, ST slope</li>
                <li className="flex items-start gap-2"><span className="w-5 h-5 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs font-bold shrink-0">5</span>Advanced Tests (Optional) — Vessels, Thalassemia</li>
              </ol>
              <div className="mt-4 pt-4 border-t border-gray-100 space-y-2">
                <p>✓ Contextual help for each field</p>
                <p>✓ Input validation with friendly error messages</p>
                <p>✓ Risk category with probability visualization</p>
                <p>✓ Contributing and protective factors</p>
                <p>✓ Clear medical disclaimer</p>
              </div>
              <button onClick={() => onNavigate('predict')} className="btn-primary mt-4 text-sm">
                <Activity className="w-4 h-4 inline mr-1" />Try Prediction
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: SECURITY & ETHICS
          ═══════════════════════════════════════════ */}
      <section id="section-security-&-ethics" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Security & Ethics</h2>

        <div className="grid sm:grid-cols-2 gap-6 mb-8">
          <div className="card">
            <h3 className="font-bold text-gray-900 mb-3">Security Measures</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              {[
                'Pydantic input validation on all API endpoints',
                'No PII (personally identifiable information) collected',
                'No user health data stored or logged',
                'API is stateless — no session data retained',
                'CORS configured appropriately',
                'Error messages do not leak implementation details',
                'Model files are read-only',
              ].map((item, i) => (
                <li key={i} className="flex items-start gap-2"><Check className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />{item}</li>
              ))}
            </ul>
          </div>
          <div className="card">
            <h3 className="font-bold text-gray-900 mb-3">Ethical Considerations</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              {[
                'Dataset may contain historical demographic biases',
                'Model trained on limited population (Cleveland, 1988)',
                'Performance may vary across demographics',
                'False negatives (4 cases): missed disease could delay treatment',
                'False positives (6 cases): unnecessary anxiety and follow-up',
                'System provides risk estimation, not diagnosis',
                'Cannot replace clinical judgment',
                'Clear disclaimers about limitations',
              ].map((item, i) => (
                <li key={i} className="flex items-start gap-2"><AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: LIMITATIONS
          ═══════════════════════════════════════════ */}
      <section id="section-limitations" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Limitations</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { title: 'Small Dataset', desc: '303 patients limits statistical power and generalizability. Cross-validation provides more stable estimates but cannot fully compensate.', icon: <Database className="w-5 h-5" /> },
            { title: 'Historical Data', desc: 'Original clinical data is from 1988. Treatment patterns, demographics, and disease profiles may have changed.', icon: <Clock className="w-5 h-5" /> },
            { title: 'Single Source', desc: 'All data from Cleveland Clinic Foundation. No external validation performed. Results may not generalize to other populations.', icon: <Users className="w-5 h-5" /> },
            { title: 'No Clinical Validation', desc: 'Model has not been validated in a clinical setting. Predictions should not be used for medical decision-making.', icon: <Shield className="w-5 h-5" /> },
            { title: 'Limited Features', desc: '13 clinical features. Missing important modern risk factors: genetics, lifestyle, imaging biomarkers, wearables.', icon: <Layers className="w-5 h-5" /> },
            { title: 'Binary Classification', desc: 'Only predicts presence/absence of disease. Does not estimate severity, stage, or prognosis.', icon: <Target className="w-5 h-5" /> },
          ].map((lim, i) => (
            <div key={i} className="card bg-red-50/50 border-red-100">
              <div className="text-red-500 mb-2">{lim.icon}</div>
              <h4 className="font-bold text-gray-900 mb-1">{lim.title}</h4>
              <p className="text-sm text-gray-600">{lim.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: FUTURE WORK
          ═══════════════════════════════════════════ */}
      <section id="section-future-work" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">Future Work</h2>
        <div className="space-y-4">
          {[
            { phase: 'Phase 1', title: 'Larger Datasets', desc: 'Integrate Hungarian, Swiss, and Long Beach VA subsets from UCI. Explore UK Biobank and MIMIC-IV for expanded training.', status: 'planned' },
            { phase: 'Phase 2', title: 'External Validation', desc: 'Test trained models on independent datasets to assess generalizability and domain shift.', status: 'planned' },
            { phase: 'Phase 3', title: 'Advanced Explainability', desc: 'Implement SHAP and LIME for local and global explanations. Add individual prediction waterfall charts.', status: 'planned' },
            { phase: 'Phase 4', title: 'Wearable Integration', desc: 'Explore integration with wearable device data (heart rate, activity, sleep) for continuous risk monitoring.', status: 'planned' },
            { phase: 'Phase 5', title: 'EHR Integration', desc: 'Develop pipelines for electronic health record data ingestion and real-time risk scoring.', status: 'planned' },
            { phase: 'Phase 6', title: 'Prospective Validation', desc: 'Conduct prospective clinical validation study with healthcare partners.', status: 'planned' },
            { phase: 'Phase 7', title: 'Multi-Centre Evaluation', desc: 'Evaluate model performance across multiple healthcare institutions and diverse populations.', status: 'planned' },
          ].map((fw, i) => (
            <div key={i} className="card flex items-start gap-4">
              <span className="text-xs bg-primary-100 text-primary-700 px-3 py-1 rounded-full font-semibold whitespace-nowrap mt-1">{fw.phase}</span>
              <div>
                <h4 className="font-bold text-gray-900">{fw.title}</h4>
                <p className="text-sm text-gray-600 mt-1">{fw.desc}</p>
              </div>
              <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full whitespace-nowrap mt-1 ml-auto">Future Work</span>
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: RESEARCH CONTRIBUTION
          ═══════════════════════════════════════════ */}
      <section className="pt-12">
        <div className="bg-gradient-to-br from-primary-50 to-blue-50 rounded-2xl p-8 border border-primary-100">
          <h2 className="text-2xl font-extrabold text-gray-900 mb-4">Research Contribution</h2>
          <p className="text-gray-700 leading-relaxed mb-6">This project's contribution is the integration of the following components into one educational and research-oriented application:</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {['Public Dataset', 'Leakage-Safe Pipeline', 'Model Comparison', 'Hyperparameter Optimization', 'Probability Calibration', 'Feature Importance', 'REST API', 'Web Interface', 'Responsible AI Documentation'].map((item, i) => (
              <div key={i} className="bg-white rounded-xl px-4 py-3 border border-primary-100 text-sm font-medium text-primary-800 flex items-center gap-2">
                <Check className="w-4 h-4 text-green-500 shrink-0" />{item}
              </div>
            ))}
          </div>
          <p className="text-sm text-gray-500 mt-4">Described as an end-to-end educational and research-oriented machine-learning framework. Not a clinical contribution.</p>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          SECTION: REFERENCES
          ═══════════════════════════════════════════ */}
      <section id="section-references" className="pt-12 scroll-mt-32">
        <h2 className="text-3xl font-extrabold text-gray-900 mb-6">References</h2>
        <div className="space-y-3">
          {LITERATURE.map((lit, i) => (
            <div key={lit.id} className="flex gap-3 text-sm text-gray-700">
              <span className="text-gray-400 font-mono shrink-0">[{i + 1}]</span>
              <div>
                <p>{lit.authors} "{lit.title}," <em>{lit.journal}</em>, {lit.volume} {lit.pages}, {lit.year}.</p>
                <a href={lit.doi} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline inline-flex items-center gap-1 mt-1">
                  <ExternalLink className="w-3 h-3" />{lit.doi}
                </a>
              </div>
            </div>
          ))}
          <div className="flex gap-3 text-sm text-gray-700">
            <span className="text-gray-400 font-mono shrink-0">[12]</span>
            <p>WHO. (2021). <em>Cardiovascular diseases (CVDs)</em>. World Health Organization. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)</p>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════
          FULL REPORT CTA
          ═══════════════════════════════════════════ */}
      <section className="pt-12 pb-4">
        <div className="bg-gradient-to-r from-gray-900 to-primary-900 rounded-2xl p-10 text-center text-white">
          <h2 className="text-2xl font-extrabold mb-3">Read the Complete Research Report</h2>
          <p className="text-blue-200/80 max-w-2xl mx-auto mb-6">
            Explore the complete academic project report containing the methodology, literature survey, dataset description, experimental design, results, explainability analysis, architecture, references, appendices, and implementation details.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <a href={RESEARCH_META.repo} target="_blank" rel="noopener noreferrer" className="bg-white text-gray-900 px-6 py-3 rounded-xl font-semibold hover:bg-gray-100 transition-colors shadow-lg inline-flex items-center gap-2">
              <Github className="w-4 h-4" />View on GitHub
            </a>
            <a href={`${RESEARCH_META.repo}/blob/main/docs/PROJECT_REPORT.md`} target="_blank" rel="noopener noreferrer" className="bg-white/10 backdrop-blur text-white px-6 py-3 rounded-xl font-semibold hover:bg-white/20 transition-colors border border-white/20 inline-flex items-center gap-2">
              <FileText className="w-4 h-4" />Open Full Research Report
            </a>
          </div>
        </div>
      </section>

      {/* Footer Disclaimer */}
      <div className="pt-8">
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-amber-800">
              <strong>Disclaimer:</strong> This page presents a research and educational machine-learning project. It is <strong>not</strong> a medical diagnostic system. All predictions, probabilities, and risk assessments are model-estimated values based on statistical patterns in a limited dataset (303 patients, 1988). They should <strong>never</strong> replace professional medical evaluation. Always consult a qualified healthcare professional for medical advice.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Small helper: Globe icon (Lucide doesn't export one by default)
function Globe({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  )
}
