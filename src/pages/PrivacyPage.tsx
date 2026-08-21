import { ArrowLeft, Shield, Eye, Database, Lock, Globe, UserX, Mail, FileText } from 'lucide-react'
import type { Page } from '../App'

interface Props { onNavigate: (page: Page) => void }

export default function PrivacyPage({ onNavigate }: Props) {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={() => onNavigate('home')} className="p-2 rounded-xl hover:bg-gray-100 transition-colors">
          <ArrowLeft className="w-5 h-5 text-gray-600" />
        </button>
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-gray-900">Privacy Policy</h1>
          <p className="text-sm text-gray-500 mt-1">Last updated: August 2026</p>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Introduction */}
        <Section icon={<Shield className="w-5 h-5" />} title="1. Introduction">
          <p>This Privacy Policy explains how the HeartGuard ML application ("the Application") handles information. We are committed to protecting your privacy and being transparent about our practices.</p>
          <p>This Application was developed by <strong>Souvik Barui</strong> as a research and educational project for heart disease risk prediction using machine learning.</p>
        </Section>

        {/* Data Collection */}
        <Section icon={<Database className="w-5 h-5" />} title="2. Data We Collect">
          <div className="bg-green-50 border border-green-200 rounded-xl p-4 mb-4">
            <p className="text-sm text-green-800 font-medium">✓ Good News: We collect minimal data</p>
          </div>
          <p>The Application collects the following health-related input data <strong>only for the purpose of generating predictions</strong>:</p>
          <ul className="list-disc pl-5 space-y-1 mt-2">
            <li>Age</li>
            <li>Biological sex</li>
            <li>Chest pain type</li>
            <li>Resting blood pressure</li>
            <li>Serum cholesterol</li>
            <li>Fasting blood sugar</li>
            <li>Resting ECG results</li>
            <li>Maximum heart rate</li>
            <li>Exercise-induced angina</li>
            <li>ST depression</li>
            <li>ST segment slope</li>
            <li>Number of major vessels</li>
            <li>Thalassemia status</li>
          </ul>
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mt-4">
            <p className="text-sm text-amber-800"><strong>Important:</strong> This data is processed in real-time and is <strong>never stored, logged, or transmitted</strong> to any external server beyond the API processing.</p>
          </div>
        </Section>

        {/* Data Usage */}
        <Section icon={<Eye className="w-5 h-5" />} title="3. How We Use Your Data">
          <p>Input data is used <strong>solely</strong> for:</p>
          <ul className="list-disc pl-5 space-y-1 mt-2">
            <li>Generating heart disease risk predictions</li>
            <li>Providing feature importance explanations</li>
            <li>Calculating probability scores</li>
          </ul>
          <p className="mt-2">Data is processed immediately and discarded after the prediction is returned. No data persistence occurs.</p>
        </Section>

        {/* What We Don't Do */}
        <Section icon={<UserX className="w-5 h-5 text-red-500" />} title="4. What We Don't Do">
          <p>We want to be completely transparent about what we do <strong>NOT</strong> do with your data:</p>
          <ul className="list-disc pl-5 space-y-1 mt-2">
            <li><strong>Do NOT</strong> store your health information</li>
            <li><strong>Do NOT</strong> share your data with third parties</li>
            <li><strong>Do NOT</strong> use your data for marketing or advertising</li>
            <li><strong>Do NOT</strong> sell your data to anyone</li>
            <li><strong>Do NOT</strong> track your browsing behavior</li>
            <li><strong>Do NOT</strong> use cookies for tracking (only essential PWA functionality)</li>
            <li><strong>Do NOT</strong> collect personal identifiers (name, email, phone)</li>
            <li><strong>Do NOT</strong> create user profiles</li>
          </ul>
        </Section>

        {/* Data Security */}
        <Section icon={<Lock className="w-5 h-5" />} title="5. Data Security">
          <p>We implement the following security measures:</p>
          <ul className="list-disc pl-5 space-y-1 mt-2">
            <li>Input validation using Pydantic schemas</li>
            <li>No data persistence in the database</li>
            <li>Stateless API design</li>
            <li>CORS configuration</li>
            <li>Error messages that don't leak implementation details</li>
            <li>Client-side processing for the web interface</li>
          </ul>
        </Section>

        {/* Third-Party Services */}
        <Section icon={<Globe className="w-5 h-5" />} title="6. Third-Party Services">
          <p>The Application uses the following third-party services:</p>
          <ul className="list-disc pl-5 space-y-1 mt-2">
            <li><strong>GitHub:</strong> For source code hosting (no user data transmitted)</li>
            <li><strong>Google Fonts:</strong> For typography (standard CDN, no tracking)</li>
          </ul>
          <p className="mt-2">We do not integrate with any analytics, advertising, or tracking services.</p>
        </Section>

        {/* Children's Privacy */}
        <Section icon={<Shield className="w-5 h-5" />} title="7. Children's Privacy">
          <p>This Application is not directed at children under 13. We do not knowingly collect personal information from children. If you are a parent or guardian and believe your child has provided us with personal information, please contact us.</p>
        </Section>

        {/* Changes */}
        <Section icon={<FileText className="w-5 h-5" />} title="8. Changes to This Policy">
          <p>We may update this Privacy Policy from time to time. Changes will be posted on this page with an updated revision date. Your continued use of the Application after changes constitutes acceptance of the new policy.</p>
        </Section>

        {/* Contact */}
        <Section icon={<Mail className="w-5 h-5" />} title="9. Contact Us">
          <p>If you have questions about this Privacy Policy, please contact:</p>
          <div className="bg-gray-50 rounded-xl p-4 mt-2">
            <p className="font-semibold text-gray-800">Souvik Barui</p>
            <p className="text-sm text-gray-600">Research & Development</p>
            <p className="text-sm text-gray-600 mt-1">Email: <a href="mailto:projectmakersb@gmail.com" className="text-primary-600 hover:underline">projectmakersb@gmail.com</a></p>
            <p className="text-sm text-gray-600 mt-1">GitHub: <a href="https://github.com/souvikbarui2003" target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">github.com/souvikbarui2003</a></p>
          </div>
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
