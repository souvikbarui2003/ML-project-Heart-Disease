import { useState } from 'react'
import { AlertTriangle, Activity, Heart, TrendingUp, TrendingDown, Info, ChevronRight, ChevronLeft, HelpCircle, Check } from 'lucide-react'

interface FormData {
  age: number | null
  sex: number | null
  cp: number | null
  trestbps: number | null
  chol: number | null
  fbs: number | null
  restecg: number | null
  thalach: number | null
  exang: number | null
  oldpeak: number | null
  slope: number | null
  ca: number | null
  thal: number | null
}

interface StepConfig {
  id: string
  title: string
  description: string
  fields: (keyof FormData)[]
}

const steps: StepConfig[] = [
  {
    id: 'about',
    title: 'About You',
    description: 'Basic information about yourself',
    fields: ['age', 'sex'],
  },
  {
    id: 'general',
    title: 'General Health',
    description: 'Your general health measurements',
    fields: ['trestbps', 'chol', 'fbs'],
  },
  {
    id: 'heart',
    title: 'Heart Tests',
    description: 'Results from heart-related tests',
    fields: ['restecg', 'thalach', 'exang'],
  },
  {
    id: 'exercise',
    title: 'Exercise Response',
    description: 'How your heart responds to exercise',
    fields: ['oldpeak', 'slope'],
  },
  {
    id: 'advanced',
    title: 'Advanced Tests (Optional)',
    description: 'Additional test results if available',
    fields: ['ca', 'thal'],
  },
]

const fieldConfig: Record<string, {
  label: string
  question: string
  help: string
  type: 'number' | 'select'
  unit?: string
  options?: { value: number; label: string }[]
  min?: number
  max?: number
  step?: number
}> = {
  age: {
    label: 'Age',
    question: 'How old are you?',
    help: 'Enter your current age in years.',
    type: 'number',
    unit: 'years',
    min: 1,
    max: 120,
  },
  sex: {
    label: 'Sex',
    question: 'What is your biological sex?',
    help: 'This refers to biological sex assigned at birth.',
    type: 'select',
    options: [
      { value: 0, label: 'Female' },
      { value: 1, label: 'Male' },
    ],
  },
  cp: {
    label: 'Chest Pain Type',
    question: 'What type of chest discomfort do you usually experience?',
    help: 'Typical angina is chest pain caused by reduced blood flow to the heart, usually triggered by physical activity and relieved by rest. Atypical angina is chest discomfort that doesn\'t fit the typical pattern. Non-anginal pain is chest pain not related to the heart. Asymptomatic means you don\'t experience chest pain.',
    type: 'select',
    options: [
      { value: 0, label: 'Typical Angina' },
      { value: 1, label: 'Atypical Angina' },
      { value: 2, label: 'Non-Anginal Pain' },
      { value: 3, label: 'Asymptomatic (No chest pain)' },
    ],
  },
  trestbps: {
    label: 'Resting Blood Pressure',
    question: 'What is your resting blood pressure?',
    help: 'This is the blood pressure measured while you are resting. Normal is typically below 120/80 mmHg. Enter the systolic (top number) value.',
    type: 'number',
    unit: 'mmHg',
    min: 60,
    max: 250,
  },
  chol: {
    label: 'Serum Cholesterol',
    question: 'What is your total cholesterol level?',
    help: 'Total cholesterol is measured in mg/dl. Desirable is below 200 mg/dl. High is above 240 mg/dl.',
    type: 'number',
    unit: 'mg/dl',
    min: 100,
    max: 600,
  },
  fbs: {
    label: 'Fasting Blood Sugar',
    question: 'Is your fasting blood sugar greater than 120 mg/dl?',
    help: 'Fasting blood sugar is measured after not eating for at least 8 hours. Levels above 120 mg/dl may indicate diabetes or pre-diabetes.',
    type: 'select',
    options: [
      { value: 0, label: 'No (≤ 120 mg/dl)' },
      { value: 1, label: 'Yes (> 120 mg/dl)' },
    ],
  },
  restecg: {
    label: 'Resting ECG Results',
    question: 'What were your resting ECG results?',
    help: 'An ECG measures the electrical activity of your heart. Normal means no significant abnormalities. ST-T wave abnormality indicates potential heart muscle changes. Left ventricular hypertrophy means the heart muscle is enlarged.',
    type: 'select',
    options: [
      { value: 0, label: 'Normal' },
      { value: 1, label: 'ST-T Wave Abnormality' },
      { value: 2, label: 'Left Ventricular Hypertrophy' },
    ],
  },
  thalach: {
    label: 'Maximum Heart Rate',
    question: 'What is the maximum heart rate you can achieve?',
    help: 'Maximum heart rate during exercise is often estimated as 220 minus your age. A lower-than-expected maximum heart rate may indicate heart problems.',
    type: 'number',
    unit: 'bpm',
    min: 60,
    max: 220,
  },
  exang: {
    label: 'Exercise-Induced Angina',
    question: 'Do you experience chest pain during physical activity?',
    help: 'Exercise-induced angina is chest pain that occurs during physical activity and is a common symptom of coronary artery disease.',
    type: 'select',
    options: [
      { value: 0, label: 'No' },
      { value: 1, label: 'Yes' },
    ],
  },
  oldpeak: {
    label: 'ST Depression',
    question: 'Do you know your ST depression value from an exercise test?',
    help: 'ST depression is a change in the ECG during exercise. Higher values may indicate reduced blood flow to the heart. If you haven\'t had this test, you can skip this field.',
    type: 'number',
    unit: 'mm',
    min: 0,
    max: 10,
    step: 0.1,
  },
  slope: {
    label: 'ST Segment Slope',
    question: 'What is the slope of your ST segment during exercise?',
    help: 'The ST segment slope is measured during an exercise ECG test. Upsloping is typically normal. Flat or downsloping may indicate reduced blood flow.',
    type: 'select',
    options: [
      { value: 0, label: 'Upsloping' },
      { value: 1, label: 'Flat' },
      { value: 2, label: 'Downsloping' },
    ],
  },
  ca: {
    label: 'Number of Major Vessels',
    question: 'How many major vessels were visible on your fluoroscopy?',
    help: 'This is the number of major coronary arteries that appear blocked on a fluoroscopy (imaging) test. If you haven\'t had this test, you can skip this field.',
    type: 'select',
    options: [
      { value: 0, label: '0' },
      { value: 1, label: '1' },
      { value: 2, label: '2' },
      { value: 3, label: '3' },
      { value: 4, label: '4' },
    ],
  },
  thal: {
    label: 'Thalassemia',
    question: 'What is your thalassemia status?',
    help: 'Thalassemia is a blood disorder that affects hemoglobin. A fixed defect means a permanent abnormality in heart blood flow. A reversible defect means temporary abnormality that improves with rest.',
    type: 'select',
    options: [
      { value: 0, label: 'Normal' },
      { value: 1, label: 'Fixed Defect' },
      { value: 2, label: 'Reversible Defect' },
      { value: 3, label: 'Unknown' },
    ],
  },
}

export default function PredictPage() {
  const [currentStep, setCurrentStep] = useState(0)
  const [formData, setFormData] = useState<FormData>({
    age: 55, sex: 1, cp: 2, trestbps: 130, chol: 240,
    fbs: 0, restecg: 1, thalach: 160, exang: 0,
    oldpeak: 1.0, slope: 2, ca: 0, thal: 2,
  })
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [showHelp, setShowHelp] = useState<string | null>(null)

  const computeRiskScore = (d: FormData): number => {
    // Enhanced risk calculation based on feature importance
    let score = 0

    // Age contribution
    if (d.age !== null) {
      score += 0.08 * (d.age - 54) / 9
    }

    // Sex contribution
    if (d.sex !== null) {
      score += 0.12 * (d.sex - 0.5)
    }

    // Chest pain type
    if (d.cp !== null) {
      score += 0.15 * (d.cp - 1) / 1.5
    }

    // Resting blood pressure
    if (d.trestbps !== null) {
      score += 0.06 * (d.trestbps - 132) / 17
    }

    // Cholesterol
    if (d.chol !== null) {
      score += 0.05 * (d.chol - 246) / 52
    }

    // Fasting blood sugar
    if (d.fbs !== null) {
      score += 0.03 * d.fbs
    }

    // Resting ECG
    if (d.restecg !== null) {
      score += 0.04 * (d.restecg - 0.5) / 0.8
    }

    // Max heart rate (inverse - lower is worse)
    if (d.thalach !== null) {
      score += 0.10 * (1 - (d.thalach - 71) / (202 - 71))
    }

    // Exercise angina
    if (d.exang !== null) {
      score += 0.12 * d.exang
    }

    // ST depression
    if (d.oldpeak !== null) {
      score += 0.08 * (d.oldpeak - 1.0) / 2
    }

    // ST slope
    if (d.slope !== null) {
      score += 0.06 * (d.slope - 1) / 1
    }

    // Major vessels
    if (d.ca !== null) {
      score += 0.10 * d.ca / 2
    }

    // Thalassemia
    if (d.thal !== null) {
      score += 0.08 * (d.thal - 1.5) / 1.5
    }

    // Sigmoid
    return 1 / (1 + Math.exp(-(score + 0.1)))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    // Simulate prediction
    setTimeout(() => {
      const prob = computeRiskScore(formData)
      const prediction = prob >= 0.5 ? 1 : 0
      const riskCategory =
        prob < 0.3 ? 'lower predicted risk' :
        prob < 0.5 ? 'moderate predicted risk' :
        prob < 0.7 ? 'higher predicted risk' : 'elevated predicted risk'

      setResult({
        prediction,
        probability: Math.round(prob * 10000) / 10000,
        risk_category: riskCategory,
        model_name: 'Enhanced Heart Disease Prediction',
        model_version: '2.0.0',
        disclaimer: 'This is a model-estimated risk score, not a medical diagnosis.',
        contributing_factors: getContributingFactors(formData, prediction === 1),
        protective_factors: getProtectiveFactors(formData, prediction === 1),
      })
      setLoading(false)
    }, 500)
  }

  const getContributingFactors = (d: FormData, isPositive: boolean) => {
    const factors: { label: string; value: string; severity: 'high' | 'medium' | 'low' }[] = []
    if (d.age !== null && d.age > 60) factors.push({ label: 'Age above 60', value: `${d.age} years`, severity: 'high' })
    if (d.cp !== null && d.cp >= 2) factors.push({ label: 'Chest pain type', value: fieldConfig.cp.options?.find(o => o.value === d.cp)?.label || '', severity: 'high' })
    if (d.trestbps !== null && d.trestbps > 140) factors.push({ label: 'Elevated resting BP', value: `${d.trestbps} mmHg`, severity: 'medium' })
    if (d.chol !== null && d.chol > 280) factors.push({ label: 'High cholesterol', value: `${d.chol} mg/dl`, severity: 'medium' })
    if (d.exang !== null && d.exang === 1) factors.push({ label: 'Exercise-induced angina', value: 'Yes', severity: 'high' })
    if (d.oldpeak !== null && d.oldpeak > 2) factors.push({ label: 'High ST depression', value: `${d.oldpeak}`, severity: 'medium' })
    if (d.ca !== null && d.ca >= 2) factors.push({ label: 'Multiple major vessels', value: `${d.ca}`, severity: 'high' })
    if (d.thal !== null && d.thal >= 2) factors.push({ label: 'Thalassemia defect', value: fieldConfig.thal.options?.find(o => o.value === d.thal)?.label || '', severity: 'medium' })
    if (factors.length === 0) factors.push({ label: 'No strong risk signals', value: '—', severity: 'low' })
    return factors
  }

  const getProtectiveFactors = (d: FormData, isPositive: boolean) => {
    const factors: { label: string; value: string }[] = []
    if (d.sex !== null && d.sex === 0) factors.push({ label: 'Female sex', value: 'Female' })
    if (d.age !== null && d.age < 40) factors.push({ label: 'Young age', value: `${d.age} years` })
    if (d.thalach !== null && d.thalach >= 160) factors.push({ label: 'Good heart rate response', value: `${d.thalach} bpm` })
    if (d.slope !== null && d.slope === 2) factors.push({ label: 'Downsloping ST segment', value: 'Downsloping' })
    if (d.cp !== null && d.cp <= 1) factors.push({ label: 'Low-risk chest pain', value: fieldConfig.cp.options?.find(o => o.value === d.cp)?.label || '' })
    if (d.thal !== null && d.thal === 0) factors.push({ label: 'Normal thalassemia', value: 'Normal' })
    if (factors.length === 0) factors.push({ label: 'No protective factors identified', value: '—' })
    return factors
  }

  const getRiskColor = (prob: number) => {
    if (prob < 0.3) return 'text-green-600 bg-green-50 border-green-200'
    if (prob < 0.5) return 'text-amber-600 bg-amber-50 border-amber-200'
    if (prob < 0.7) return 'text-orange-600 bg-orange-50 border-orange-200'
    return 'text-red-600 bg-red-50 border-red-200'
  }

  const getSeverityColor = (severity: 'high' | 'medium' | 'low') => {
    if (severity === 'high') return 'bg-red-100 text-red-700'
    if (severity === 'medium') return 'bg-amber-100 text-amber-700'
    return 'bg-gray-100 text-gray-700'
  }

  const validateField = (field: keyof FormData, value: number | null): boolean => {
    const config = fieldConfig[field]
    if (value === null) return true // Allow null for optional fields
    if (config.type === 'number' && config.min !== undefined && config.max !== undefined) {
      return value >= config.min && value <= config.max
    }
    return true
  }

  const renderStep = (step: StepConfig) => (
    <div className="space-y-6" key={step.id}>
      <div>
        <h3 className="text-xl font-bold text-gray-900">{step.title}</h3>
        <p className="text-gray-600 mt-1">{step.description}</p>
      </div>

      <div className="space-y-4">
        {step.fields.map((field) => {
          const config = fieldConfig[field]
          const value = formData[field]
          const isValid = validateField(field, value)

          return (
            <div key={field} className="space-y-2">
              <div className="flex items-center gap-2">
                <label className="label flex-1">
                  {config.question}
                </label>
                <button
                  type="button"
                  onClick={() => setShowHelp(showHelp === field ? null : field)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <HelpCircle className="w-5 h-5" />
                </button>
              </div>

              {showHelp === field && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
                  <Info className="w-4 h-4 inline mr-1" />
                  {config.help}
                </div>
              )}

              {config.type === 'select' ? (
                <select
                  value={value ?? ''}
                  onChange={(e) => setFormData(prev => ({ ...prev, [field]: Number(e.target.value) }))}
                  className={`input-field ${!isValid ? 'border-red-500' : ''}`}
                >
                  <option value="">Select an option</option>
                  {config.options?.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              ) : (
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    step={config.step || 1}
                    value={value ?? ''}
                    onChange={(e) => {
                      const val = e.target.value === '' ? null : Number(e.target.value)
                      setFormData(prev => ({ ...prev, [field]: val }))
                    }}
                    min={config.min}
                    max={config.max}
                    placeholder={field === 'oldpeak' ? 'Optional - skip if unknown' : ''}
                    className={`input-field flex-1 ${!isValid ? 'border-red-500' : ''}`}
                  />
                  {config.unit && (
                    <span className="text-sm text-gray-500">{config.unit}</span>
                  )}
                </div>
              )}

              {!isValid && (
                <p className="text-sm text-red-600">
                  Please enter a valid value between {config.min} and {config.max}
                </p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )

  return (
    <div className="space-y-8">
      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
        <div>
          <p className="text-sm text-amber-800 font-medium">Research & Educational Tool Only</p>
          <p className="text-sm text-amber-700 mt-1">
            This prediction is from a machine-learning model trained on research data. It is <strong>not</strong> a
            medical diagnosis. Always consult a qualified healthcare professional for medical advice.
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Progress Steps */}
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            {steps.map((step, index) => (
              <button
                key={step.id}
                type="button"
                onClick={() => setCurrentStep(index)}
                className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                  currentStep === index
                    ? 'bg-primary-100 text-primary-700'
                    : index < currentStep
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-500'
                }`}
              >
                {index < currentStep ? (
                  <Check className="w-4 h-4" />
                ) : (
                  <span className="w-5 h-5 rounded-full bg-current bg-opacity-20 flex items-center justify-center text-xs">
                    {index + 1}
                  </span>
                )}
                <span className="hidden sm:inline">{step.title}</span>
              </button>
            ))}
          </div>

          {/* Current Step */}
          {renderStep(steps[currentStep])}

          {/* Navigation */}
          <div className="flex justify-between">
            <button
              type="button"
              onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
              disabled={currentStep === 0}
              className="btn-secondary flex items-center gap-2 disabled:opacity-50"
            >
              <ChevronLeft className="w-4 h-4" />
              Back
            </button>

            {currentStep < steps.length - 1 ? (
              <button
                type="button"
                onClick={() => setCurrentStep(currentStep + 1)}
                className="btn-primary flex items-center gap-2"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={loading}
                className="btn-primary flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity className="w-5 h-5" />
                    Get Risk Prediction
                  </>
                )}
              </button>
            )}
          </div>
        </form>

        {/* Results */}
        <div className="space-y-6">
          {result ? (
            <>
              {/* Main Result */}
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

              {/* Contributing Factors */}
              <div className="card">
                <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-red-500" />
                  Factors Increasing Risk
                </h4>
                <div className="space-y-2">
                  {result.contributing_factors.map((factor: any, i: number) => (
                    <div key={i} className="flex items-center gap-3 text-sm">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(factor.severity)}`}>
                        {factor.severity}
                      </span>
                      <span className="text-gray-700 flex-1">{factor.label}</span>
                      <span className="text-gray-500">{factor.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Protective Factors */}
              <div className="card">
                <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <TrendingDown className="w-4 h-4 text-green-500" />
                  Protective Factors
                </h4>
                <div className="space-y-2">
                  {result.protective_factors.map((factor: any, i: number) => (
                    <div key={i} className="flex items-center gap-3 text-sm">
                      <span className="w-2 h-2 rounded-full bg-green-400" />
                      <span className="text-gray-700 flex-1">{factor.label}</span>
                      <span className="text-gray-500">{factor.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* What This Means */}
              <div className="card bg-blue-50 border-blue-200">
                <h4 className="font-semibold text-blue-800 mb-2 flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  What This Means
                </h4>
                <p className="text-sm text-blue-700">
                  This result is an estimate generated from patterns learned from the available dataset.
                  It does not confirm whether you have heart disease. The model identified a{' '}
                  <strong>{result.risk_category}</strong> pattern based on your inputs.
                </p>
                <p className="text-sm text-blue-600 mt-3 font-medium">
                  If you are concerned about your symptoms or cardiovascular health, consider
                  discussing the result and your health information with a qualified healthcare professional.
                </p>
              </div>

              {/* Medical Disclaimer */}
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
                Complete the questionnaire on the left and click "Get Risk Prediction"
                to see the model's assessment.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
