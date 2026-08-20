import { BarChart3, TrendingUp, Activity, Target, Clock, Shield } from 'lucide-react'

// Model comparison data (from actual training results)
const modelResults = [
  { name: 'Logistic Regression', cvAuc: 0.9109, testAuc: 0.8885, accuracy: 0.8361, recall: 0.8788, specificity: 0.7857, f1: 0.8571, time: 0.11, best: true },
  { name: 'SVM', cvAuc: 0.9102, testAuc: 0.8864, accuracy: 0.7869, recall: 0.8788, specificity: 0.6786, f1: 0.8235, time: 0.49, best: false },
  { name: 'K-Nearest Neighbors', cvAuc: 0.8953, testAuc: 0.8555, accuracy: 0.7541, recall: 0.8485, specificity: 0.6429, f1: 0.8000, time: 0.20, best: false },
  { name: 'Random Forest', cvAuc: 0.8882, testAuc: 0.8463, accuracy: 0.7213, recall: 0.8182, specificity: 0.6071, f1: 0.7692, time: 19.99, best: false },
  { name: 'Gradient Boosting', cvAuc: 0.8737, testAuc: 0.8506, accuracy: 0.7377, recall: 0.8182, specificity: 0.6429, f1: 0.7813, time: 27.82, best: false },
  { name: 'Decision Tree', cvAuc: 0.8274, testAuc: 0.7495, accuracy: 0.7377, recall: 0.8485, specificity: 0.6071, f1: 0.7925, time: 0.46, best: false },
]

const featureImportance = [
  { feature: 'exang (exercise angina)', importance: 0.142 },
  { feature: 'thal (thalassemia)', importance: 0.128 },
  { feature: 'ca (major vessels)', importance: 0.124 },
  { feature: 'cp (chest pain type)', importance: 0.118 },
  { feature: 'oldpeak (ST depression)', importance: 0.109 },
  { feature: 'thalach (max heart rate)', importance: 0.098 },
  { feature: 'slope (ST segment)', importance: 0.087 },
  { feature: 'sex', importance: 0.065 },
  { feature: 'age', importance: 0.052 },
  { feature: 'trestbps (resting BP)', importance: 0.035 },
]

const bestModel = modelResults[0]

export default function DashboardPage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
          <BarChart3 className="w-8 h-8 text-primary-600" />
          Model Dashboard
        </h1>
        <p className="text-gray-600 mt-2">
          Performance metrics from the trained heart disease prediction models.
          All models trained on 303 patients with 5-fold stratified cross-validation.
        </p>
      </div>

      {/* Best Model Summary */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-2xl p-8 text-white">
        <div className="flex items-center gap-2 text-primary-200 text-sm font-medium mb-2">
          <Shield className="w-4 h-4" /> Best Selected Model
        </div>
        <h2 className="text-2xl font-bold mb-4">{bestModel.name}</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
          <MetricBox label="CV ROC-AUC" value={bestModel.cvAuc} format="pct" />
          <MetricBox label="Test ROC-AUC" value={bestModel.testAuc} format="pct" />
          <MetricBox label="Accuracy" value={bestModel.accuracy} format="pct" />
          <MetricBox label="Recall (Sensitivity)" value={bestModel.recall} format="pct" />
        </div>
        <p className="text-sm text-primary-200 mt-4">
          Selected by highest CV ROC-AUC score. Logistic Regression was chosen for its strong
          performance, high interpretability, and fast inference.
        </p>
      </div>

      {/* Model Comparison Table */}
      <div className="card overflow-hidden">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-primary-600" />
          Model Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-semibold text-gray-700">Model</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">CV AUC</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Test AUC</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Accuracy</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Recall</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Specificity</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">F1</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Time</th>
              </tr>
            </thead>
            <tbody>
              {modelResults.map(m => (
                <tr key={m.name} className={`border-b border-gray-100 ${m.best ? 'bg-primary-50' : 'hover:bg-gray-50'}`}>
                  <td className="py-3 px-4 font-medium">
                    {m.name}
                    {m.best && <span className="ml-2 text-xs bg-primary-100 text-primary-700 px-2 py-0.5 rounded-full">Best</span>}
                  </td>
                  <td className="py-3 px-4 text-right font-mono">{(m.cvAuc * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right font-mono">{(m.testAuc * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right font-mono">{(m.accuracy * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right font-mono">{(m.recall * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right font-mono">{(m.specificity * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right font-mono">{(m.f1 * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right font-mono">{m.time.toFixed(2)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bar Chart Visualization */}
      <div className="card">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-primary-600" />
          CV ROC-AUC Comparison
        </h3>
        <div className="space-y-3">
          {modelResults.map(m => (
            <div key={m.name} className="flex items-center gap-3">
              <span className="w-44 text-sm text-gray-700 text-right truncate">{m.name}</span>
              <div className="flex-1 bg-gray-100 rounded-full h-7 overflow-hidden">
                <div
                  className={`h-full rounded-full flex items-center justify-end pr-3 text-xs font-bold text-white transition-all duration-700 ${
                    m.best ? 'bg-gradient-to-r from-primary-500 to-primary-700' : 'bg-gray-400'
                  }`}
                  style={{ width: `${m.cvAuc * 100}%` }}
                >
                  {(m.cvAuc * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Importance */}
      <div className="card">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-primary-600" />
          Feature Importance (Random Forest)
        </h3>
        <div className="space-y-2.5">
          {featureImportance.map(f => (
            <div key={f.feature} className="flex items-center gap-3">
              <span className="w-52 text-sm text-gray-700 text-right truncate">{f.feature}</span>
              <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-blue-400 to-primary-600 flex items-center justify-end pr-2"
                  style={{ width: `${(f.importance / 0.15) * 100}%` }}
                >
                  <span className="text-xs font-bold text-white">{(f.importance * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Healthcare Metrics Explainer */}
      <div className="card">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary-600" />
          Healthcare-Relevant Metrics Explainer
        </h3>
        <div className="grid sm:grid-cols-2 gap-4 text-sm">
          <div className="bg-blue-50 rounded-xl p-4">
            <h4 className="font-semibold text-blue-800 mb-1">Recall (Sensitivity)</h4>
            <p className="text-blue-700">The proportion of actual disease cases correctly identified. High recall means fewer missed cases — critical for screening.</p>
            <p className="text-blue-600 font-bold mt-2">Best model: {(bestModel.recall * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-green-50 rounded-xl p-4">
            <h4 className="font-semibold text-green-800 mb-1">Specificity</h4>
            <p className="text-green-700">The proportion of healthy cases correctly identified. Lower false positives means fewer unnecessary follow-ups.</p>
            <p className="text-green-600 font-bold mt-2">Best model: {(bestModel.specificity * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-purple-50 rounded-xl p-4">
            <h4 className="font-semibold text-purple-800 mb-1">ROC-AUC</h4>
            <p className="text-purple-700">Overall discriminative ability across all classification thresholds. 1.0 = perfect, 0.5 = random.</p>
            <p className="text-purple-600 font-bold mt-2">Best model: {(bestModel.testAuc * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-amber-50 rounded-xl p-4">
            <h4 className="font-semibold text-amber-800 mb-1">F1 Score</h4>
            <p className="text-amber-700">Harmonic mean of precision and recall. Balances the trade-off between catching cases and avoiding false alarms.</p>
            <p className="text-amber-600 font-bold mt-2">Best model: {(bestModel.f1 * 100).toFixed(1)}%</p>
          </div>
        </div>
      </div>

      {/* Training Info */}
      <div className="card bg-gray-50">
        <h3 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
          <Clock className="w-5 h-5 text-primary-600" />
          Training Configuration
        </h3>
        <div className="grid sm:grid-cols-2 gap-4 text-sm text-gray-700">
          <div>
            <p><strong>Dataset:</strong> 302 patients (1 duplicate removed)</p>
            <p><strong>Split:</strong> 80% train / 20% test, stratified</p>
            <p><strong>Cross-validation:</strong> 5-fold stratified</p>
            <p><strong>Random seed:</strong> 42</p>
          </div>
          <div>
            <p><strong>Preprocessing:</strong> Median imputation + StandardScaler (numerical), Mode imputation + OneHotEncoder (categorical)</p>
            <p><strong>Leakage prevention:</strong> Split before preprocessing, fit only on training data</p>
            <p><strong>Features after encoding:</strong> 22</p>
          </div>
        </div>
      </div>

      {/* Confusion Matrix for Best Model */}
      <div className="card">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Confusion Matrix — Logistic Regression</h3>
        <div className="flex justify-center">
          <div className="inline-grid grid-cols-3 gap-0 text-center text-sm">
            <div />
            <div className="font-semibold text-gray-600 pb-2">Predicted No Disease</div>
            <div className="font-semibold text-gray-600 pb-2">Predicted Disease</div>
            <div className="font-semibold text-gray-600 pr-4 pt-2 text-right">Actual No Disease</div>
            <div className="w-24 h-16 bg-green-100 rounded-lg flex flex-col items-center justify-center border border-green-200">
              <span className="text-xl font-bold text-green-700">22</span>
              <span className="text-xs text-green-600">TN</span>
            </div>
            <div className="w-24 h-16 bg-red-100 rounded-lg flex flex-col items-center justify-center border border-red-200">
              <span className="text-xl font-bold text-red-700">6</span>
              <span className="text-xs text-red-600">FP</span>
            </div>
            <div className="font-semibold text-gray-600 pr-4 pt-2 text-right">Actual Disease</div>
            <div className="w-24 h-16 bg-amber-100 rounded-lg flex flex-col items-center justify-center border border-amber-200">
              <span className="text-xl font-bold text-amber-700">4</span>
              <span className="text-xs text-amber-600">FN</span>
            </div>
            <div className="w-24 h-16 bg-blue-100 rounded-lg flex flex-col items-center justify-center border border-blue-200">
              <span className="text-xl font-bold text-blue-700">29</span>
              <span className="text-xs text-blue-600">TP</span>
            </div>
          </div>
        </div>
        <p className="text-xs text-gray-500 text-center mt-3">
          TN=True Negative, FP=False Positive, FN=False Negative, TP=True Positive
        </p>
      </div>
    </div>
  )
}

function MetricBox({ label, value, format }: { label: string; value: number; format: string }) {
  return (
    <div>
      <p className="text-primary-200 text-sm">{label}</p>
      <p className="text-2xl font-bold">
        {format === 'pct' ? `${(value * 100).toFixed(2)}%` : value.toFixed(4)}
      </p>
    </div>
  )
}
