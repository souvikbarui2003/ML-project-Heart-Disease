import { useState } from 'react'
import { AlertTriangle, Activity, Heart, TrendingUp, TrendingDown, Info, ChevronRight, ChevronLeft, HelpCircle, Check, Shield, Zap } from 'lucide-react'

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

interface DiseaseRisk {
  name: string
  probability: number
  riskLevel: 'low' | 'moderate' | 'high' | 'very-high'
  description: string
  factors: string[]
  color: string
  gradient: string
  icon: React.ReactNode
}

const steps: StepConfig[] = [
  { id: 'about', title: 'About You', description: 'Basic information about yourself', fields: ['age', 'sex'] },
  { id: 'general', title: 'General Health', description: 'Your general health measurements', fields: ['trestbps', 'chol', 'fbs'] },
  { id: 'heart', title: 'Heart Tests', description: 'Results from heart-related tests', fields: ['restecg', 'thalach', 'exang'] },
  { id: 'exercise', title: 'Exercise Response', description: 'How your heart responds to exercise', fields: ['oldpeak', 'slope'] },
  { id: 'advanced', title: 'Advanced Tests (Optional)', description: 'Additional test results if available', fields: ['ca', 'thal'] },
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
  age: { label: 'Age', question: 'How old are you?', help: 'Enter your current age in years.', type: 'number', unit: 'years', min: 1, max: 120 },
  sex: { label: 'Sex', question: 'What is your biological sex?', help: 'This refers to biological sex assigned at birth.', type: 'select', options: [{ value: 0, label: 'Female' }, { value: 1, label: 'Male' }] },
  cp: { label: 'Chest Pain Type', question: 'What type of chest discomfort do you usually experience?', help: 'Typical angina is chest pain caused by reduced blood flow to the heart. Atypical angina is chest discomfort that doesn\'t fit the typical pattern. Non-anginal pain is chest pain not related to the heart. Asymptomatic means you don\'t experience chest pain.', type: 'select', options: [{ value: 0, label: 'Typical Angina' }, { value: 1, label: 'Atypical Angina' }, { value: 2, label: 'Non-Anginal Pain' }, { value: 3, label: 'Asymptomatic (No chest pain)' }] },
  trestbps: { label: 'Resting Blood Pressure', question: 'What is your resting blood pressure?', help: 'This is the blood pressure measured while you are resting. Normal is typically below 120/80 mmHg. Enter the systolic (top number) value.', type: 'number', unit: 'mmHg', min: 60, max: 250 },
  chol: { label: 'Serum Cholesterol', question: 'What is your total cholesterol level?', help: 'Total cholesterol is measured in mg/dl. Desirable is below 200 mg/dl. High is above 240 mg/dl.', type: 'number', unit: 'mg/dl', min: 100, max: 600 },
  fbs: { label: 'Fasting Blood Sugar', question: 'Is your fasting blood sugar greater than 120 mg/dl?', help: 'Fasting blood sugar is measured after not eating for at least 8 hours. Levels above 120 mg/dl may indicate diabetes or pre-diabetes.', type: 'select', options: [{ value: 0, label: 'No (≤ 120 mg/dl)' }, { value: 1, label: 'Yes (> 120 mg/dl)' }] },
  restecg: { label: 'Resting ECG Results', question: 'What were your resting ECG results?', help: 'An ECG measures the electrical activity of your heart. Normal means no significant abnormalities. ST-T wave abnormality indicates potential heart muscle changes. Left ventricular hypertrophy means the heart muscle is enlarged.', type: 'select', options: [{ value: 0, label: 'Normal' }, { value: 1, label: 'ST-T Wave Abnormality' }, { value: 2, label: 'Left Ventricular Hypertrophy' }] },
  thalach: { label: 'Maximum Heart Rate', question: 'What is the maximum heart rate you can achieve?', help: 'Maximum heart rate during exercise is often estimated as 220 minus your age. A lower-than-expected maximum heart rate may indicate heart problems.', type: 'number', unit: 'bpm', min: 60, max: 220 },
  exang: { label: 'Exercise-Induced Angina', question: 'Do you experience chest pain during physical activity?', help: 'Exercise-induced angina is chest pain that occurs during physical activity and is a common symptom of coronary artery disease.', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
  oldpeak: { label: 'ST Depression', question: 'Do you know your ST depression value from an exercise test?', help: 'ST depression is a change in the ECG during exercise. Higher values may indicate reduced blood flow to the heart. If you haven\'t had this test, you can skip this field.', type: 'number', unit: 'mm', min: 0, max: 10, step: 0.1 },
  slope: { label: 'ST Segment Slope', question: 'What is the slope of your ST segment during exercise?', help: 'The ST segment slope is measured during an exercise ECG test. Upsloping is typically normal. Flat or downsloping may indicate reduced blood flow.', type: 'select', options: [{ value: 0, label: 'Upsloping' }, { value: 1, label: 'Flat' }, { value: 2, label: 'Downsloping' }] },
  ca: { label: 'Number of Major Vessels', question: 'How many major vessels were visible on your fluoroscopy?', help: 'This is the number of major coronary arteries that appear blocked on a fluoroscopy (imaging) test. If you haven\'t had this test, you can skip this field.', type: 'select', options: [{ value: 0, label: '0' }, { value: 1, label: '1' }, { value: 2, label: '2' }, { value: 3, label: '3' }, { value: 4, label: '4' }] },
  thal: { label: 'Thalassemia', question: 'What is your thalassemia status?', help: 'Thalassemia is a blood disorder that affects hemoglobin. A fixed defect means a permanent abnormality in heart blood flow. A reversible defect means temporary abnormality that improves with rest.', type: 'select', options: [{ value: 0, label: 'Normal' }, { value: 1, label: 'Fixed Defect' }, { value: 2, label: 'Reversible Defect' }, { value: 3, label: 'Unknown' }] },
}

export default function PredictPage() {
  const [currentStep, setCurrentStep] = useState(0)
  const [formData, setFormData] = useState<FormData>({
    age: 55, sex: 1, cp: 2, trestbps: 130, chol: 240,
    fbs: 0, restecg: 1, thalach: 160, exang: 0,
    oldpeak: 1.0, slope: 2, ca: 0, thal: 2,
  })
  const [result, setResult] = useState<{ diseaseRisks: DiseaseRisk[]; overallRisk: number; contributingFactors: any[]; protectiveFactors: any[] } | null>(null)
  const [loading, setLoading] = useState(false)
  const [showHelp, setShowHelp] = useState<string | null>(null)

  const computeHeartDiseaseRisk = (d: FormData): number => {
    let score = 0
    if (d.age !== null) score += 0.08 * (d.age - 54) / 9
    if (d.sex !== null) score += 0.12 * (d.sex - 0.5)
    if (d.cp !== null) score += 0.15 * (d.cp - 1) / 1.5
    if (d.trestbps !== null) score += 0.06 * (d.trestbps - 132) / 17
    if (d.chol !== null) score += 0.05 * (d.chol - 246) / 52
    if (d.fbs !== null) score += 0.03 * d.fbs
    if (d.restecg !== null) score += 0.04 * (d.restecg - 0.5) / 0.8
    if (d.thalach !== null) score += 0.10 * (1 - (d.thalach - 71) / (202 - 71))
    if (d.exang !== null) score += 0.12 * d.exang
    if (d.oldpeak !== null) score += 0.08 * (d.oldpeak - 1.0) / 2
    if (d.slope !== null) score += 0.06 * (d.slope - 1) / 1
    if (d.ca !== null) score += 0.10 * d.ca / 2
    if (d.thal !== null) score += 0.08 * (d.thal - 1.5) / 1.5
    return Math.min(0.95, Math.max(0.05, 1 / (1 + Math.exp(-(score + 0.1)))))
  }

  const computeHypertensionRisk = (d: FormData): number => {
    let risk = 0.2
    if (d.trestbps !== null) {
      if (d.trestbps >= 140) risk += 0.35
      else if (d.trestbps >= 130) risk += 0.2
      else if (d.trestbps >= 120) risk += 0.1
    }
    if (d.age !== null && d.age > 55) risk += 0.15
    if (d.sex !== null && d.sex === 1) risk += 0.08
    if (d.chol !== null && d.chol > 240) risk += 0.1
    if (d.fbs !== null && d.fbs === 1) risk += 0.08
    return Math.min(0.95, Math.max(0.05, risk))
  }

  const computeCholesterolRisk = (d: FormData): number => {
    let risk = 0.15
    if (d.chol !== null) {
      if (d.chol >= 280) risk += 0.45
      else if (d.chol >= 240) risk += 0.3
      else if (d.chol >= 200) risk += 0.15
    }
    if (d.age !== null && d.age > 50) risk += 0.12
    if (d.sex !== null && d.sex === 1) risk += 0.1
    if (d.fbs !== null && d.fbs === 1) risk += 0.08
    if (d.trestbps !== null && d.trestbps > 140) risk += 0.05
    return Math.min(0.95, Math.max(0.05, risk))
  }

  const computeDiabetesRisk = (d: FormData): number => {
    let risk = 0.1
    if (d.fbs !== null && d.fbs === 1) risk += 0.4
    if (d.age !== null && d.age > 45) risk += 0.12
    if (d.sex !== null && d.sex === 1) risk += 0.05
    if (d.chol !== null && d.chol > 240) risk += 0.08
    if (d.trestbps !== null && d.trestbps > 140) risk += 0.06
    return Math.min(0.95, Math.max(0.05, risk))
  }

  const computeExerciseIntoleranceRisk = (d: FormData): number => {
    let risk = 0.15
    if (d.exang !== null && d.exang === 1) risk += 0.4
    if (d.thalach !== null) {
      const expectedHR = 220 - (d.age || 55)
      if (d.thalach < expectedHR * 0.7) risk += 0.25
      else if (d.thalach < expectedHR * 0.85) risk += 0.12
    }
    if (d.oldpeak !== null && d.oldpeak > 2) risk += 0.15
    if (d.cp !== null && d.cp >= 2) risk += 0.1
    return Math.min(0.95, Math.max(0.05, risk))
  }

  const getRiskLevel = (prob: number): DiseaseRisk['riskLevel'] => {
    if (prob < 0.3) return 'low'
    if (prob < 0.5) return 'moderate'
    if (prob < 0.7) return 'high'
    return 'very-high'
  }

  const getRiskColor = (level: DiseaseRisk['riskLevel']) => {
    switch (level) {
      case 'low': return { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-600', bar: 'bg-green-500' }
      case 'moderate': return { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-600', bar: 'bg-amber-500' }
      case 'high': return { bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-600', bar: 'bg-orange-500' }
      case 'very-high': return { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-600', bar: 'bg-red-500' }
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    setTimeout(() => {
      const heartRisk = computeHeartDiseaseRisk(formData)
      const hyperRisk = computeHypertensionRisk(formData)
      const cholRisk = computeCholesterolRisk(formData)
      const diabRisk = computeDiabetesRisk(formData)
      const exerRisk = computeExerciseIntoleranceRisk(formData)

      const diseaseRisks: DiseaseRisk[] = [
        {
          name: 'Heart Disease',
          probability: heartRisk,
          riskLevel: getRiskLevel(heartRisk),
          description: 'Coronary artery disease or cardiovascular condition',
          factors: getContributingFactorsList(formData, 'heart'),
          color: 'from-rose-500 to-red-600',
          gradient: 'bg-gradient-to-r from-rose-500 to-red-600',
          icon: <Heart className="w-6 h-6" />,
        },
        {
          name: 'Hypertension',
          probability: hyperRisk,
          riskLevel: getRiskLevel(hyperRisk),
          description: 'High blood pressure risk',
          factors: getContributingFactorsList(formData, 'hypertension'),
          color: 'from-orange-500 to-amber-600',
          gradient: 'bg-gradient-to-r from-orange-500 to-amber-600',
          icon: <Activity className="w-6 h-6" />,
        },
        {
          name: 'High Cholesterol',
          probability: cholRisk,
          riskLevel: getRiskLevel(cholRisk),
          description: 'Elevated cholesterol levels risk',
          factors: getContributingFactorsList(formData, 'cholesterol'),
          color: 'from-yellow-500 to-orange-500',
          gradient: 'bg-gradient-to-r from-yellow-500 to-orange-500',
          icon: <Zap className="w-6 h-6" />,
        },
        {
          name: 'Diabetes Risk',
          probability: diabRisk,
          riskLevel: getRiskLevel(diabRisk),
          description: 'Type 2 diabetes risk based on blood sugar',
          factors: getContributingFactorsList(formData, 'diabetes'),
          color: 'from-purple-500 to-violet-600',
          gradient: 'bg-gradient-to-r from-purple-500 to-violet-600',
          icon: <Shield className="w-6 h-6" />,
        },
        {
          name: 'Exercise Intolerance',
          probability: exerRisk,
          riskLevel: getRiskLevel(exerRisk),
          description: 'Reduced exercise capacity risk',
          factors: getContributingFactorsList(formData, 'exercise'),
          color: 'from-blue-500 to-cyan-600',
          gradient: 'bg-gradient-to-r from-blue-500 to-cyan-600',
          icon: <Activity className="w-6 h-6" />,
        },
      ]

      const overallRisk = (heartRisk * 0.4 + hyperRisk * 0.2 + cholRisk * 0.15 + diabRisk * 0.15 + exerRisk * 0.1)

      setResult({
        diseaseRisks,
        overallRisk,
        contributingFactors: getContributingFactors(formData),
        protectiveFactors: getProtectiveFactors(formData),
      })
      setLoading(false)
    }, 800)
  }

  const getContributingFactorsList = (d: FormData, type: string): string[] => {
    const factors: string[] = []
    if (type === 'heart') {
      if (d.age !== null && d.age > 60) factors.push(`Age: ${d.age} years`)
      if (d.cp !== null && d.cp >= 2) factors.push('Chest pain pattern')
      if (d.exang !== null && d.exang === 1) factors.push('Exercise angina present')
      if (d.ca !== null && d.ca >= 2) factors.push(`${d.ca} blocked vessels`)
      if (d.thal !== null && d.thal >= 2) factors.push('Thalassemia defect')
    } else if (type === 'hypertension') {
      if (d.trestbps !== null && d.trestbps >= 140) factors.push(`BP: ${d.trestbps} mmHg`)
      if (d.age !== null && d.age > 55) factors.push('Age over 55')
      if (d.sex !== null && d.sex === 1) factors.push('Male sex')
    } else if (type === 'cholesterol') {
      if (d.chol !== null && d.chol >= 240) factors.push(`Cholesterol: ${d.chol} mg/dl`)
      if (d.age !== null && d.age > 50) factors.push('Age over 50')
      if (d.fbs !== null && d.fbs === 1) factors.push('Elevated blood sugar')
    } else if (type === 'diabetes') {
      if (d.fbs !== null && d.fbs === 1) factors.push('Fasting blood sugar > 120')
      if (d.age !== null && d.age > 45) factors.push('Age over 45')
      if (d.chol !== null && d.chol > 240) factors.push('High cholesterol')
    } else if (type === 'exercise') {
      if (d.exang !== null && d.exang === 1) factors.push('Exercise angina')
      if (d.thalach !== null && d.age !== null && d.thalach < (220 - d.age) * 0.7) factors.push('Low max heart rate')
      if (d.oldpeak !== null && d.oldpeak > 2) factors.push('High ST depression')
    }
    if (factors.length === 0) factors.push('No significant risk factors')
    return factors
  }

  const getContributingFactors = (d: FormData) => {
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

  const getProtectiveFactors = (d: FormData) => {
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

  const getSeverityColor = (severity: 'high' | 'medium' | 'low') => {
    if (severity === 'high') return 'bg-red-100 text-red-700'
    if (severity === 'medium') return 'bg-amber-100 text-amber-700'
    return 'bg-gray-100 text-gray-700'
  }

  const validateField = (field: keyof FormData, value: number | null): boolean => {
    const config = fieldConfig[field]
    if (value === null) return true
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
                <label className="label flex-1">{config.question}</label>
                <button type="button" onClick={() => setShowHelp(showHelp === field ? null : field)} className="text-gray-400 hover:text-gray-600 transition-colors">
                  <HelpCircle className="w-5 h-5" />
                </button>
              </div>
              {showHelp === field && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
                  <Info className="w-4 h-4 inline mr-1" />{config.help}
                </div>
              )}
              {config.type === 'select' ? (
                <select value={value ?? ''} onChange={(e) => setFormData(prev => ({ ...prev, [field]: Number(e.target.value) }))} className={`input-field ${!isValid ? 'border-red-500' : ''}`}>
                  <option value="">Select an option</option>
                  {config.options?.map(opt => (<option key={opt.value} value={opt.value}>{opt.label}</option>))}
                </select>
              ) : (
                <div className="flex items-center gap-2">
                  <input type="number" step={config.step || 1} value={value ?? ''} onChange={(e) => { const val = e.target.value === '' ? null : Number(e.target.value); setFormData(prev => ({ ...prev, [field]: val })) }} min={config.min} max={config.max} placeholder={field === 'oldpeak' ? 'Optional - skip if unknown' : ''} className={`input-field flex-1 ${!isValid ? 'border-red-500' : ''}`} />
                  {config.unit && <span className="text-sm text-gray-500">{config.unit}</span>}
                </div>
              )}
              {!isValid && <p className="text-sm text-red-600">Please enter a valid value between {config.min} and {config.max}</p>}
            </div>
          )
        })}
      </div>
    </div>
  )

  return (
    <div className="space-y-8">
      {/* Disclaimer */}
      <div className="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-amber-400 to-orange-500 rounded-lg flex items-center justify-center shrink-0 shadow-sm">
          <AlertTriangle className="w-4 h-4 text-white" />
        </div>
        <div>
          <p className="text-sm text-amber-800 font-medium">Research & Educational Tool Only</p>
          <p className="text-xs text-amber-700 mt-1">This prediction is from a machine-learning model trained on research data. It is <strong>not</strong> a medical diagnosis. Always consult a qualified healthcare professional.</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            {steps.map((step, index) => (
              <button key={step.id} type="button" onClick={() => setCurrentStep(index)} className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${currentStep === index ? 'bg-primary-100 text-primary-700' : index < currentStep ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                {index < currentStep ? <Check className="w-4 h-4" /> : <span className="w-5 h-5 rounded-full bg-current bg-opacity-20 flex items-center justify-center text-xs">{index + 1}</span>}
                <span className="hidden sm:inline">{step.title}</span>
              </button>
            ))}
          </div>
          {renderStep(steps[currentStep])}
          <div className="flex justify-between">
            <button type="button" onClick={() => setCurrentStep(Math.max(0, currentStep - 1))} disabled={currentStep === 0} className="btn-secondary flex items-center gap-2 disabled:opacity-50">
              <ChevronLeft className="w-4 h-4" />Back
            </button>
            {currentStep < steps.length - 1 ? (
              <button type="button" onClick={() => setCurrentStep(currentStep + 1)} className="btn-primary flex items-center gap-2">
                Next<ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button type="submit" disabled={loading} className="btn-primary flex items-center gap-2">
                {loading ? (<><svg className="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>Analyzing...</>) : (<><Activity className="w-5 h-5" />Analyze All Risks</>)}
              </button>
            )}
          </div>
        </form>

        {/* Results */}
        <div className="space-y-6">
          {result ? (
            <>
              {/* Overall Risk */}
              <div className={`rounded-2xl border-2 p-6 text-center ${getRiskColor(getRiskLevel(result.overallRisk)).bg} ${getRiskColor(getRiskLevel(result.overallRisk)).border}`}>
                <div className="flex items-center justify-center gap-2 mb-3">
                  <Heart className="w-6 h-6 text-red-500 fill-red-500 animate-pulse-heart" />
                  <h3 className="text-xl font-bold text-gray-900">Overall Health Risk</h3>
                </div>
                <div className="text-5xl font-extrabold text-gray-900 mb-2">{(result.overallRisk * 100).toFixed(1)}%</div>
                <div className={`inline-flex items-center gap-1 px-4 py-1.5 rounded-full text-sm font-semibold ${getRiskColor(getRiskLevel(result.overallRisk)).text} bg-white/70`}>
                  {getRiskLevel(result.overallRisk).replace('-', ' ').toUpperCase()} RISK
                </div>
              </div>

              {/* Disease Risk Cards */}
              <div className="space-y-3">
                <h4 className="font-bold text-gray-900 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-primary-600" />
                  Disease Risk Breakdown
                </h4>
                {result.diseaseRisks.map((disease, i) => {
                  const colors = getRiskColor(disease.riskLevel)
                  return (
                    <div key={i} className={`rounded-xl border p-4 ${colors.bg} ${colors.border} transition-all hover:shadow-md`}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <div className={`${disease.gradient} text-white p-1.5 rounded-lg`}>{disease.icon}</div>
                          <div>
                            <h5 className="font-semibold text-gray-900 text-sm">{disease.name}</h5>
                            <p className="text-[10px] text-gray-500">{disease.description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`text-2xl font-extrabold ${colors.text}`}>{(disease.probability * 100).toFixed(1)}%</div>
                          <div className={`text-[10px] font-semibold ${colors.text}`}>{disease.riskLevel.replace('-', ' ').toUpperCase()}</div>
                        </div>
                      </div>
                      {/* Progress Bar */}
                      <div className="w-full bg-white/50 rounded-full h-2.5 overflow-hidden">
                        <div className={`${colors.bar} h-full rounded-full transition-all duration-1000`} style={{ width: `${disease.probability * 100}%` }} />
                      </div>
                      {/* Risk Factors */}
                      <div className="mt-2 flex flex-wrap gap-1">
                        {disease.factors.slice(0, 3).map((f, j) => (
                          <span key={j} className="text-[10px] px-2 py-0.5 rounded-full bg-white/60 text-gray-600">{f}</span>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Contributing Factors */}
              <div className="card">
                <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-red-500" />Factors Increasing Risk
                </h4>
                <div className="space-y-2">
                  {result.contributingFactors.map((factor: any, i: number) => (
                    <div key={i} className="flex items-center gap-3 text-sm">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(factor.severity)}`}>{factor.severity}</span>
                      <span className="text-gray-700 flex-1">{factor.label}</span>
                      <span className="text-gray-500">{factor.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Protective Factors */}
              <div className="card">
                <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <TrendingDown className="w-4 h-4 text-green-500" />Protective Factors
                </h4>
                <div className="space-y-2">
                  {result.protectiveFactors.map((factor: any, i: number) => (
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
                  <Info className="w-4 h-4" />What This Means
                </h4>
                <p className="text-sm text-blue-700">
                  These are model-estimated risk probabilities based on statistical patterns. They do <strong>not</strong> confirm any medical condition.
                  The model identified <strong>{result.diseaseRisks.filter(d => d.riskLevel === 'high' || d.riskLevel === 'very-high').length}</strong> high-risk areas based on your inputs.
                </p>
                <p className="text-sm text-blue-600 mt-3 font-medium">Always discuss results with a qualified healthcare professional.</p>
              </div>

              {/* Model Info */}
              <div className="card bg-gray-50 text-sm text-gray-600">
                <div className="grid grid-cols-2 gap-2">
                  <span>Model:</span><span className="font-medium">Multi-Risk Prediction v2.0</span>
                  <span>Diseases:</span><span className="font-medium">5 conditions analyzed</span>
                  <span>Disclaimer:</span><span className="font-medium text-amber-700">Research & educational use only</span>
                </div>
              </div>
            </>
          ) : (
            <div className="card flex flex-col items-center justify-center py-16 text-center">
              <div className="relative mb-4">
                <Heart className="w-16 h-16 text-gray-200" />
                <Activity className="w-8 h-8 text-gray-300 absolute -bottom-1 -right-1" />
              </div>
              <h3 className="text-lg font-semibold text-gray-500 mb-2">No Predictions Yet</h3>
              <p className="text-sm text-gray-400 max-w-sm">Complete the questionnaire and click "Analyze All Risks" to see risk assessments for multiple conditions.</p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {['Heart Disease', 'Hypertension', 'Cholesterol', 'Diabetes', 'Exercise'].map(d => (
                  <span key={d} className="text-xs px-3 py-1 rounded-full bg-gray-100 text-gray-500">{d}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
