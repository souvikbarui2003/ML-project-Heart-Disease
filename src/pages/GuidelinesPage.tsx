import { ArrowLeft, BookOpen, AlertTriangle, CheckCircle, Heart, Activity, BarChart3, Lightbulb, Shield, Target } from 'lucide-react'
import type { Page } from '../App'

interface Props { onNavigate: (page: Page) => void }

export default function GuidelinesPage({ onNavigate }: Props) {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={() => onNavigate('home')} className="p-2 rounded-xl hover:bg-gray-100 transition-colors">
          <ArrowLeft className="w-5 h-5 text-gray-600" />
        </button>
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-gray-900">Usage Guidelines</h1>
          <p className="text-sm text-gray-500 mt-1">How to use HeartGuard ML effectively and responsibly</p>
        </div>
      </div>

      {/* Quick Start */}
      <div className="card bg-gradient-to-br from-rose-50 to-orange-50 border-rose-200">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-gradient-to-br from-rose-400 to-orange-500 rounded-xl flex items-center justify-center shadow-md">
            <Heart className="w-5 h-5 text-white fill-white" />
          </div>
          <h2 className="text-xl font-bold text-gray-900">Quick Start Guide</h2>
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          <Step number={1} title="Enter Information" description="Fill in the clinical questionnaire with your health measurements" />
          <Step number={2} title="Get Prediction" description="Receive a calibrated risk probability with explanations" />
          <Step number={3} title="Understand Results" description="Review contributing factors and protective factors" />
        </div>
      </div>

      {/* Guidelines */}
      <div className="space-y-6">
        <Section icon={<Activity className="w-5 h-5" />} title="1. How to Use the Prediction Tool">
          <h3 className="font-semibold text-gray-800 mt-3 mb-2">Step-by-Step Process:</h3>
          <ol className="list-decimal pl-5 space-y-2">
            <li><strong>Navigate to the Predict page</strong> using the navigation menu</li>
            <li><strong>Fill in the questionnaire</strong> with your clinical measurements</li>
            <li><strong>Use the help icons</strong> (?) next to each field for explanations</li>
            <li><strong>Skip unknown fields</strong> — the model handles missing data</li>
            <li><strong>Click "Get Risk Prediction"</strong> to generate results</li>
            <li><strong>Review the output</strong> including risk category and contributing factors</li>
          </ol>
        </Section>

        <Section icon={<Target className="w-5 h-5" />} title="2. Understanding Your Results">
          <h3 className="font-semibold text-gray-800 mt-3 mb-2">Risk Categories:</h3>
          <div className="grid sm:grid-cols-3 gap-3 mt-3">
            <div className="bg-green-50 border border-green-200 rounded-xl p-3 text-center">
              <div className="text-green-600 font-bold">Lower Risk</div>
              <div className="text-xs text-green-700 mt-1">Below 30% probability</div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 text-center">
              <div className="text-amber-600 font-bold">Moderate Risk</div>
              <div className="text-xs text-amber-700 mt-1">30-50% probability</div>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-xl p-3 text-center">
              <div className="text-red-600 font-bold">Higher Risk</div>
              <div className="text-xs text-red-700 mt-1">Above 50% probability</div>
            </div>
          </div>
          <p className="mt-3 text-sm text-gray-600">The probability represents the model's estimated risk based on statistical patterns. It is <strong>not</strong> a medical diagnosis.</p>
        </Section>

        <Section icon={<Lightbulb className="w-5 h-5" />} title="3. Tips for Best Results">
          <ul className="list-disc pl-5 space-y-2 mt-2">
            <li><strong>Use actual measurements</strong> when available — don't guess</li>
            <li><strong>Be consistent</strong> with units (mmHg for BP, mg/dl for cholesterol)</li>
            <li><strong>Use recent measurements</strong> — not values from years ago</li>
            <li><strong>Don't manipulate inputs</strong> to get a preferred result</li>
            <li><strong>Review the explanation</strong> — understand what drives the prediction</li>
            <li><strong>Consult a doctor</strong> for any health concerns</li>
          </ul>
        </Section>

        <Section icon={<AlertTriangle className="w-5 h-5 text-amber-500" />} title="4. Important Limitations">
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-4">
            <p className="text-sm text-amber-800"><strong>⚠️ This is NOT a medical device.</strong></p>
          </div>
          <ul className="list-disc pl-5 space-y-2 mt-2">
            <li>Model trained on only <strong>303 patients</strong> from Cleveland (1988)</li>
            <li>May not generalize to all populations or demographics</li>
            <li>Missing important risk factors (genetics, lifestyle, imaging)</li>
            <li>Binary classification only (no severity levels)</li>
            <li>Not validated in clinical settings</li>
            <li>Predictions are statistical estimates, not diagnoses</li>
          </ul>
        </Section>

        <Section icon={<Shield className="w-5 h-5" />} title="5. Medical Disclaimer">
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-sm text-red-800 leading-relaxed">
              <strong>This application is a research and educational tool.</strong> It is <strong>not</strong> a medical
              diagnostic device. Predictions should not replace evaluation by a qualified healthcare
              professional. Always consult a physician for medical advice. If you are experiencing
              chest pain, shortness of breath, or other symptoms, seek immediate medical attention.
            </p>
          </div>
        </Section>

        <Section icon={<BarChart3 className="w-5 h-5" />} title="6. Exploring the Dashboard">
          <p>The Dashboard provides detailed model performance metrics:</p>
          <ul className="list-disc pl-5 space-y-2 mt-2">
            <li><strong>Model Comparison:</strong> See how different algorithms perform</li>
            <li><strong>Feature Importance:</strong> Understand which features matter most</li>
            <li><strong>Dataset Info:</strong> Learn about the training data</li>
            <li><strong>Training Configuration:</strong> See the ML pipeline details</li>
            <li><strong>Metrics Explainer:</strong> Understand what each metric means</li>
          </ul>
        </Section>

        <Section icon={<BookOpen className="w-5 h-5" />} title="7. Research Documentation">
          <p>For detailed technical information, visit the Research page:</p>
          <ul className="list-disc pl-5 space-y-2 mt-2">
            <li><strong>Literature Survey:</strong> Academic references and comparisons</li>
            <li><strong>Methodology:</strong> Detailed ML pipeline description</li>
            <li><strong>Results:</strong> Comprehensive model evaluation</li>
            <li><strong>Explainability:</strong> Feature importance and prediction explanations</li>
            <li><strong>References:</strong> All cited papers with DOI links</li>
          </ul>
          <button onClick={() => onNavigate('research')} className="mt-3 text-sm text-primary-600 hover:underline font-medium">
            → Go to Research Page
          </button>
        </Section>
      </div>
    </div>
  )
}

function Section({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <div className="text-primary-600">{icon}</div>
        <h2 className="text-lg font-bold text-gray-900">{title}</h2>
      </div>
      <div className="text-sm text-gray-700 leading-relaxed space-y-2">{children}</div>
    </div>
  )
}

function Step({ number, title, description }: { number: number; title: string; description: string }) {
  return (
    <div className="flex items-start gap-3">
      <div className="w-8 h-8 bg-gradient-to-br from-rose-400 to-orange-500 rounded-full flex items-center justify-center text-white font-bold text-sm shrink-0 shadow-md">
        {number}
      </div>
      <div>
        <h3 className="font-semibold text-gray-800 text-sm">{title}</h3>
        <p className="text-xs text-gray-600 mt-0.5">{description}</p>
      </div>
    </div>
  )
}
