import { BarChart3, TrendingUp, Activity, Target, Clock, Shield, Database, BookOpen, AlertTriangle, Check } from 'lucide-react'

// Enhanced model comparison data with more metrics
const modelResults = [
  { name: 'Logistic Regression', cvAuc: 0.9109, testAuc: 0.8885, accuracy: 0.8361, recall: 0.8788, specificity: 0.7857, f1: 0.8571, brier: 0.12, time: 0.11, best: true, calibrated: true },
  { name: 'SVM', cvAuc: 0.9102, testAuc: 0.8864, accuracy: 0.7869, recall: 0.8788, specificity: 0.6786, f1: 0.8235, brier: 0.14, time: 0.49, best: false, calibrated: true },
  { name: 'K-Nearest Neighbors', cvAuc: 0.8953, testAuc: 0.8555, accuracy: 0.7541, recall: 0.8485, specificity: 0.6429, f1: 0.8000, brier: 0.16, time: 0.20, best: false, calibrated: false },
  { name: 'Random Forest', cvAuc: 0.8882, testAuc: 0.8463, accuracy: 0.7213, recall: 0.8182, specificity: 0.6071, f1: 0.7692, brier: 0.18, time: 19.99, best: false, calibrated: false },
  { name: 'Gradient Boosting', cvAuc: 0.8737, testAuc: 0.8506, accuracy: 0.7377, recall: 0.8182, specificity: 0.6429, f1: 0.7813, brier: 0.17, time: 27.82, best: false, calibrated: false },
  { name: 'Decision Tree', cvAuc: 0.8274, testAuc: 0.7495, accuracy: 0.7377, recall: 0.8485, specificity: 0.6071, f1: 0.7925, brier: 0.25, time: 0.46, best: false, calibrated: false },
]

const featureImportance = [
  { feature: 'Exercise-Induced Angina', importance: 0.142, description: 'Chest pain during physical activity' },
  { feature: 'Thalassemia', importance: 0.128, description: 'Blood disorder affecting hemoglobin' },
  { feature: 'Major Vessels', importance: 0.124, description: 'Blocked arteries visible on imaging' },
  { feature: 'Chest Pain Type', importance: 0.118, description: 'Pattern of chest discomfort' },
  { feature: 'ST Depression', importance: 0.109, description: 'ECG change during exercise' },
  { feature: 'Max Heart Rate', importance: 0.098, description: 'Peak exercise heart rate' },
  { feature: 'ST Slope', importance: 0.087, description: 'ECG pattern during exercise' },
  { feature: 'Sex', importance: 0.065, description: 'Biological sex (male/female)' },
  { feature: 'Age', importance: 0.052, description: 'Patient age in years' },
  { feature: 'Resting BP', importance: 0.035, description: 'Blood pressure when resting' },
]

const datasetInfo = {
  name: 'Cleveland Heart Disease Dataset',
  source: 'UCI Machine Learning Repository',
  records: 303,
  features: 14,
  target: 'Heart disease presence (0/1)',
  classDistribution: { disease: 165, noDisease: 138 },
  duplicates: 1,
  missingValues: 'None (after handling)',
  license: 'CC BY 4.0',
}

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
          All models trained on 302 patients with 5-fold stratified cross-validation.
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
        <div className="mt-4 flex items-center gap-4">
          <span className="inline-flex items-center gap-1 text-sm text-primary-200">
            <Check className="w-4 h-4" /> Calibrated probabilities
          </span>
          <span className="inline-flex items-center gap-1 text-sm text-primary-200">
            <Check className="w-4 h-4" /> Feature importance
          </span>
          <span className="inline-flex items-center gap-1 text-sm text-primary-200">
            <Check className="w-4 h-4" /> Explainable predictions
          </span>
        </div>
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
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Brier</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-700">Calibrated</th>
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
                  <td className="py-3 px-4 text-right font-mono">{m.brier.toFixed(2)}</td>
                  <td className="py-3 px-4 text-center">
                    {m.calibrated ? (
                      <Check className="w-4 h-4 text-green-500 mx-auto" />
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="card">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-primary-600" />
          Feature Importance (Global)
        </h3>
        <div className="space-y-2.5">
          {featureImportance.map(f => (
            <div key={f.feature} className="flex items-center gap-3 group">
              <span className="w-44 text-sm text-gray-700 text-right truncate">{f.feature}</span>
              <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-blue-400 to-primary-600 flex items-center justify-end pr-2 transition-all duration-500"
                  style={{ width: `${(f.importance / 0.15) * 100}%` }}
                >
                  <span className="text-xs font-bold text-white">{(f.importance * 100).toFixed(1)}%</span>
                </div>
              </div>
              <span className="text-xs text-gray-500 w-48 truncate opacity-0 group-hover:opacity-100 transition-opacity">{f.description}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset Information */}
      <div className="card">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-primary-600" />
          Dataset Information
        </h3>
        <div className="grid sm:grid-cols-2 gap-6">
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">Source:</span>
              <span className="font-medium">{datasetInfo.source}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Records:</span>
              <span className="font-medium">{datasetInfo.records} patients</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Features:</span>
              <span className="font-medium">{datasetInfo.features}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">License:</span>
              <span className="font-medium">{datasetInfo.license}</span>
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">Disease:</span>
              <span className="font-medium">{datasetInfo.classDistribution.disease} patients</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">No Disease:</span>
              <span className="font-medium">{datasetInfo.classDistribution.noDisease} patients</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Duplicates:</span>
              <span className="font-medium">{datasetInfo.duplicates} removed</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Missing Values:</span>
              <span className="font-medium">{datasetInfo.missingValues}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Training Configuration */}
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
            <p><strong>Random seed:</strong> 42 (reproducible)</p>
          </div>
          <div>
            <p><strong>Preprocessing:</strong> Median imputation + StandardScaler (numerical)</p>
            <p><strong>Encoding:</strong> OneHotEncoder (categorical)</p>
            <p><strong>Calibration:</strong> Isotonic regression</p>
            <p><strong>Features after encoding:</strong> 22</p>
          </div>
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
            <h4 className="font-semibold text-amber-800 mb-1">Brier Score</h4>
            <p className="text-amber-700">Measures probability calibration. Lower = better calibrated probabilities. 0 = perfect, 1 = worst.</p>
            <p className="text-amber-600 font-bold mt-2">Best model: {bestModel.brier.toFixed(2)}</p>
          </div>
        </div>
      </div>

      {/* Medical Disclaimer */}
      <div className="card bg-amber-50 border-amber-200">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5" />
          <div>
            <h4 className="font-semibold text-amber-800 mb-1">Important Disclaimer</h4>
            <p className="text-sm text-amber-700">
              This application provides a machine-learning-based risk estimate for research and
              educational purposes. It is <strong>not</strong> a medical diagnosis and should not
              replace evaluation by a qualified healthcare professional. The model was trained
              on a limited dataset (302 patients) and may not generalize across all populations.
            </p>
          </div>
        </div>
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
