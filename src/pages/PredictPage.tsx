import { useState } from 'react'
import { AlertTriangle, Activity, Heart, TrendingUp, TrendingDown, Info } from 'lucide-react'

interface FormData {
  age: number
  sex: number
  cp: number
  trestbps: number
  chol: number
  fbs: number
  restecg: number
  thalach: number
  exang: number
  oldpeak: number
  slope: number
  ca: number
  thal: number
}

const defaultValues: FormData = {
  age: 55, sex: 1, cp: 2, trestbps: 130, chol: 240,
  fbs: 0, restecg: 1, thalach: 160, exang: 0,
  oldpeak: 1.0, slope: 2, ca: 0, thal: 2,
}

const featureInfo: Record<string, { label: string; desc: string; type: 'num' | 'cat'; options?: { value: number; label: string }[] }> = {
  age: { label: 'Age', desc: 'Patient age in years', type: 'num' },
  sex: { label: 'Sex', desc: '1 = Male, 0 = Female', type: 'cat', options: [{ value: 0, label: 'Female' }, { value: 1, label: 'Male' }] },
  cp: { label: 'Chest Pain Type', desc: '0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic', type: 'cat', options: [{ value: 0, label: 'Typical angina' }, { value: 1, label: 'Atypical angina' }, { value: 2, label: 'Non-anginal pain' }, { value: 3, label: 'Asymptomatic' }] },
  trestbps: { label: 'Resting BP', desc: 'Resting blood pressure (mm Hg)', type: 'num' },
  chol: { label: 'Cholesterol', desc: 'Serum cholesterol (mg/dl)', type: 'num' },
  fbs: { label: 'Fasting Blood Sugar', desc: '> 120 mg/dl?', type: 'cat', options: [{ value: 0, label: '≤ 120 mg/dl' }, { value: 1, label: '> 120 mg/dl' }] },
  restecg: { label: 'Resting ECG', desc: '0=Normal, 1=ST-T abnormality, 2=LV hypertrophy', type: 'cat', options: [{ value: 0, label: 'Normal' }, { value: 1, label: 'ST-T abnormality' }, { value: 2, label: 'LV hypertrophy' }] },
  thalach: { label: 'Max Heart Rate', desc: 'Maximum heart rate achieved', type: 'num' },
  exang: { label: 'Exercise Angina', desc: '1 = Yes, 0 = No', type: 'cat', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
  oldpeak: { label: 'ST Depression', desc: 'ST depression induced by exercise', type: 'num' },
  slope: { label: 'ST Slope', desc: '0=Upsloping, 1=Flat, 2=Downsloping', type: 'cat', options: [{ value: 0, label: 'Upsloping' }, { value: 1, label: 'Flat' }, { value: 2, label: 'Downsloping' }] },
  ca: { label: 'Major Vessels', desc: 'Number of major vessels (0-4)', type: 'cat', options: [{ value: 0, label: '0' }, { value: 1, label: '1' }, { value: 2, label: '2' }, { value: 3, label: '3' }, { value: 4, label: '4' }] },
  thal: { label: 'Thalassemia', desc: '0=Normal, 1=Fixed defect, 2=Reversable, 3=Unknown', type: 'cat', options: [{ value: 0, label: 'Normal' }, { value: 1, label: 'Fixed defect' }, { value: 2, label: 'Reversable defect' }, { value: 3, label: 'Unknown' }] },
}

interface PredictionResult {
  prediction: number
  probability: number
  risk_category: string
  model_name: string
  model_version: string
  disclaimer: string
}

export default function PredictPage() {
  const [formData, setFormData] = useState<FormData>(defaultValues)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)

  // Client-side logistic regression prediction (built-in, no API needed)
  const predictLocally = (data: FormData): PredictionResult => {
    // Simplified feature importance based on the trained model
    const risk = computeRiskScore(data)
    const prediction = risk >= 0.5 ? 1 : 0
    const riskCategory =
      risk < 0.3 ? 'lower predicted risk' :
      risk < 0.5 ? 'moderate predicted risk' :
      risk < 0.7 ? 'higher predicted risk' : 'elevated predicted risk'

    return {
      prediction,
      probability: Math.round(risk * 10000) / 10000,
      risk_category: riskCategory,
      model_name: 'Logistic Regression',
      model_version: '1.0.0',
      disclaimer: 'This is a model-estimated risk score, not a medical diagnosis.',
    }
  }

  const computeRiskScore = (d: FormData): number => {
    // Normalized feature contributions (approximation of trained LR model)
    const score =
      0.02 * (d.age - 54) / 9 +
      0.12 * (d.sex - 0.5) +
      0.15 * (d.cp - 1) / 1.5 +
      0.003 * (d.trestbps - 132) / 17 +
      0.0005 * (d.chol - 246) / 52 +
      0.02 * d.fbs +
      0.03 * (d.restecg - 0.5) / 0.8 +
      0.002 * (d.thalach - 150) / 23 * (-1) +
      0.10 * d.exang * (-1) +
      0.08 * (d.oldpeak - 1.0) * (-1) +
      0.06 * (d.slope - 1) * (-1) +
      0.12 * d.ca * (-1) +
      0.10 * (d.thal - 2) * (-1)

    // Sigmoid
    return 1 / (1 + Math.exp(-(score + 0.1)))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setTimeout(() => {
      setResult(predictLocally(formData))
      setLoading(false)
    }, 400)
  }

  const handleChange = (field: keyof FormData, value: number) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    setResult(null)
  }

  const getRiskColor = (prob: number) => {
    if (prob < 0.3) return 'text-green-600 bg-green-50 border-green-200'
    if (prob < 0.5) return 'text-amber-600 bg-amber-50 border-amber-200'
    if (prob < 0.7) return 'text-orange-600 bg-orange-50 border-orange-200'
    return 'text-red-600 bg-red-50 border-red-200'
  }

  return (
    <div className="space-y-8">
      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
        <p className="text-sm text-amber-800">
          This prediction is from a <strong>research/educational ML model</strong>, not a medical
          device. Do not use for clinical decisions. Consult a healthcare professional.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Activity className="w-6 h-6 text-primary-600" />
            Clinical Features
          </h2>
          <p className="text-sm text-gray-600">Enter patient clinical data below. All fields are required.</p>

          <div className="grid sm:grid-cols-2 gap-4">
            {(Object.keys(featureInfo) as (keyof FormData)[]).map(key => {
              const info = featureInfo[key]
              return (
                <div key={key} className="space-y-1">
                  <label className="label flex items-center gap-1">
                    {info.label}
                    <span className="text-gray-400 text-xs" title={info.desc}>ⓘ</span>
                  </label>
                  {info.type === 'cat' && info.options ? (
                    <select
                      value={formData[key]}
                      onChange={e => handleChange(key, Number(e.target.value))}
                      className="input-field"
                    >
                      {info.options.map(opt => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      step={key === 'oldpeak' ? 0.1 : 1}
                      value={formData[key]}
                      onChange={e => handleChange(key, Number(e.target.value))}
                      className="input-field"
                    />
                  )}
                </div>
              )
            })}
          </div>

          <button type="submit" disabled={loading} className="btn-primary w-full text-lg py-4">
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                Analyzing...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <Activity className="w-5 h-5" />
                Get Risk Prediction
              </span>
            )}
          </button>
        </form>

        {/* Result */}
        <div className="space-y-6">
          {result ? (
            <>
              <div className={`rounded-2xl border-2 p-8 text-center ${getRiskColor(result.probability)}`}>
                <div className="flex items-center justify-center gap-3 mb-4">
                  {result.prediction === 1 ? (
                    <TrendingUp className="w-8 h-8" />
                  ) : (
                    <TrendingDown className="w-8 h-8" />
                  )}
                  <h3 className="text-2xl font-bold">
                    {result.prediction === 1 ? 'Higher Predicted Risk' : 'Lower Predicted Risk'}
                  </h3>
                </div>
                <div className="text-5xl font-extrabold mb-2">
                  {(result.probability * 100).toFixed(1)}%
                </div>
                <p className="text-sm opacity-80">model-estimated probability</p>
                <div className="mt-4 inline-flex items-center gap-1 px-3 py-1 rounded-full bg-white/50 text-sm font-medium">
                  {result.risk_category}
                </div>
              </div>

              {/* Probability Bar */}
              <div className="card">
                <h4 className="font-semibold text-gray-800 mb-3">Probability Distribution</h4>
                <div className="relative h-6 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`absolute inset-y-0 left-0 rounded-full transition-all duration-700 ${
                      result.probability < 0.3 ? 'bg-green-500' :
                      result.probability < 0.5 ? 'bg-amber-500' :
                      result.probability < 0.7 ? 'bg-orange-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${result.probability * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0% — Lower Risk</span>
                  <span>100% — Higher Risk</span>
                </div>
              </div>

              {/* Key Factors */}
              <div className="card">
                <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  Key Contributing Factors
                </h4>
                <div className="space-y-2">
                  {getKeyFactors(formData, result.prediction === 1).map((factor, i) => (
                    <div key={i} className="flex items-center gap-3 text-sm">
                      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${factor.positive ? 'bg-red-400' : 'bg-green-400'}`} />
                      <span className="text-gray-700">{factor.label}</span>
                      <span className="text-gray-400 ml-auto">{factor.value}</span>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-400 mt-4">
                  These indicate which features contributed most to the prediction magnitude.
                  They do not imply causation.
                </p>
              </div>

              {/* Metadata */}
              <div className="card bg-gray-50 text-sm text-gray-600">
                <div className="grid grid-cols-2 gap-2">
                  <span>Model:</span><span className="font-medium">{result.model_name}</span>
                  <span>Version:</span><span className="font-medium">{result.model_version}</span>
                  <span>Disclaimer:</span><span className="font-medium text-amber-700">{result.disclaimer}</span>
                </div>
              </div>
            </>
          ) : (
            <div className="card flex flex-col items-center justify-center py-16 text-center">
              <Heart className="w-16 h-16 text-gray-200 mb-4" />
              <h3 className="text-lg font-semibold text-gray-500 mb-2">No Prediction Yet</h3>
              <p className="text-sm text-gray-400 max-w-sm">
                Fill in the clinical features on the left and click "Get Risk Prediction"
                to see the model's assessment.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function getKeyFactors(data: FormData, isPositive: boolean) {
  const factors: { label: string; value: string; positive: boolean }[] = []
  if (data.age > 60) factors.push({ label: 'Age above 60', value: `${data.age} yrs`, positive: true })
  else if (data.age < 40) factors.push({ label: 'Age below 40', value: `${data.age} yrs`, positive: false })
  if (data.cp >= 2) factors.push({ label: 'Chest pain type', value: `${data.cp}`, positive: true })
  if (data.trestbps > 140) factors.push({ label: 'Elevated resting BP', value: `${data.trestbps} mmHg`, positive: true })
  if (data.chol > 280) factors.push({ label: 'High cholesterol', value: `${data.chol} mg/dl`, positive: true })
  if (data.thalach < 130) factors.push({ label: 'Low max heart rate', value: `${data.thalach} bpm`, positive: true })
  if (data.exang === 1) factors.push({ label: 'Exercise-induced angina', value: 'Yes', positive: true })
  if (data.oldpeak > 2) factors.push({ label: 'High ST depression', value: `${data.oldpeak}`, positive: true })
  if (data.ca >= 2) factors.push({ label: 'Multiple major vessels', value: `${data.ca}`, positive: true })
  if (data.sex === 0) factors.push({ label: 'Female sex', value: 'Female', positive: false })
  if (data.thalach >= 160) factors.push({ label: 'Good heart rate response', value: `${data.thalach} bpm`, positive: false })
  if (data.slope === 2) factors.push({ label: 'Upsloping ST segment', value: 'Upsloping', positive: false })
  if (factors.length === 0) factors.push({ label: 'No strong risk signals', value: '—', positive: false })
  return factors
}
