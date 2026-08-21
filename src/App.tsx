import { useState } from 'react'
import LandingPage from './pages/LandingPage'
import PredictPage from './pages/PredictPage'
import DashboardPage from './pages/DashboardPage'
import ResearchPage from './pages/ResearchPage'
import TermsPage from './pages/TermsPage'
import PrivacyPage from './pages/PrivacyPage'
import GuidelinesPage from './pages/GuidelinesPage'
import PwaInstallBanner from './components/PwaInstallBanner'
import { Heart, Activity, BarChart3, BookOpen, Menu, X, Shield, BookMarked, Github, FileText, ChevronRight } from 'lucide-react'

export type Page = 'home' | 'predict' | 'dashboard' | 'research' | 'terms' | 'privacy' | 'guidelines'

export default function App() {
  const [page, setPage] = useState<Page>('home')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navItems: { page: Page; icon: React.ReactNode; label: string }[] = [
    { page: 'home', icon: <Heart className="w-5 h-5" />, label: 'Home' },
    { page: 'predict', icon: <Activity className="w-5 h-5" />, label: 'Predict' },
    { page: 'dashboard', icon: <BarChart3 className="w-5 h-5" />, label: 'Dashboard' },
    { page: 'research', icon: <BookOpen className="w-5 h-5" />, label: 'Research' },
  ]

  const handleNav = (p: Page) => {
    setPage(p)
    setMobileMenuOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50">
      {/* Desktop Navigation */}
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-lg border-b border-gray-100 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14 sm:h-16">
            <button
              onClick={() => handleNav('home')}
              className="flex items-center gap-2 text-primary-700 hover:text-primary-800 transition-colors"
            >
              <Heart className="w-5 h-5 sm:w-6 sm:h-6 fill-red-500 text-red-500" />
              <span className="font-bold text-base sm:text-lg">HeartGuard ML</span>
            </button>

            {/* Desktop Nav */}
            <div className="hidden sm:flex items-center gap-1">
              {navItems.map(item => (
                <button
                  key={item.page}
                  onClick={() => handleNav(item.page)}
                  className={`flex items-center gap-1.5 px-3 lg:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    page === item.page
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}
              <div className="w-px h-6 bg-gray-200 mx-1" />
              <button onClick={() => handleNav('guidelines')} className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-50">
                <BookMarked className="w-4 h-4" /> Guide
              </button>
              <button onClick={() => handleNav('terms')} className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-50">
                <Shield className="w-4 h-4" /> T&C
              </button>
              <button onClick={() => handleNav('privacy')} className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-50">
                <FileText className="w-4 h-4" /> Privacy
              </button>
            </div>

            {/* Mobile Hamburger */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="sm:hidden p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-50 transition-colors"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Dropdown Menu */}
        {mobileMenuOpen && (
          <div className="sm:hidden border-t border-gray-100 bg-white/95 backdrop-blur-lg max-h-[80vh] overflow-y-auto">
            <div className="px-4 py-2 space-y-1">
              {/* Main Navigation */}
              {navItems.map(item => (
                <button
                  key={item.page}
                  onClick={() => handleNav(item.page)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                    page === item.page
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}

              {/* Secondary Pages */}
              <div className="border-t border-gray-100 pt-2 mt-2">
                <p className="px-4 py-1 text-xs font-semibold text-gray-400 uppercase tracking-wider">More</p>
                <button onClick={() => handleNav('guidelines')} className="w-full flex items-center justify-between px-4 py-3 rounded-xl text-sm font-medium text-gray-600 hover:bg-gray-50">
                  <div className="flex items-center gap-3">
                    <BookMarked className="w-5 h-5" />
                    Usage Guidelines
                  </div>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </button>
                <button onClick={() => handleNav('terms')} className="w-full flex items-center justify-between px-4 py-3 rounded-xl text-sm font-medium text-gray-600 hover:bg-gray-50">
                  <div className="flex items-center gap-3">
                    <Shield className="w-5 h-5" />
                    Terms & Conditions
                  </div>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </button>
                <button onClick={() => handleNav('privacy')} className="w-full flex items-center justify-between px-4 py-3 rounded-xl text-sm font-medium text-gray-600 hover:bg-gray-50">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5" />
                    Privacy Policy
                  </div>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </button>
              </div>

              {/* Developer Info */}
              <div className="border-t border-gray-100 pt-3 mt-2 pb-2">
                <div className="px-4 py-2 bg-gray-50 rounded-xl">
                  <p className="text-xs font-semibold text-gray-800">Souvik Barui</p>
                  <p className="text-[10px] text-gray-500">Research & Development</p>
                <a href="https://github.com/souvikbarui2003" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-[10px] text-primary-600 hover:underline mt-1">
                  <Github className="w-3 h-3" />GitHub Profile
                </a>
                <a href="mailto:projectmakersb@gmail.com" className="inline-flex items-center gap-1 text-[10px] text-primary-600 hover:underline mt-1">
                  ✉️ Email Developer
                </a>
                </div>
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {page === 'home' && <LandingPage onNavigate={handleNav} />}
        {page === 'predict' && <PredictPage />}
        {page === 'dashboard' && <DashboardPage />}
        {page === 'research' && <ResearchPage onNavigate={handleNav} />}
        {page === 'terms' && <TermsPage onNavigate={handleNav} />}
        {page === 'privacy' && <PrivacyPage onNavigate={handleNav} />}
        {page === 'guidelines' && <GuidelinesPage onNavigate={handleNav} />}
      </main>

      {/* PWA Install Banner */}
      <PwaInstallBanner />

      {/* Mobile Bottom Tab Bar */}
      <nav className="sm:hidden fixed bottom-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-lg border-t border-gray-200 safe-area-bottom">
        <div className="flex items-center justify-around py-2">
          {navItems.map(item => (
            <button
              key={item.page}
              onClick={() => handleNav(item.page)}
              className={`flex flex-col items-center gap-1 px-3 py-1.5 rounded-lg transition-all min-w-[60px] ${
                page === 'home'
                  ? 'text-primary-600'
                  : page === item.page
                  ? 'text-primary-600'
                  : 'text-gray-400'
              }`}
            >
              {item.icon}
              <span className="text-[10px] font-medium">{item.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Footer - Both Mobile and Desktop */}
      <footer className="border-t border-gray-100 bg-gradient-to-b from-white to-gray-50 mt-16 pb-24 sm:pb-0">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
          {/* Medical Disclaimer */}
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 sm:p-4 mb-6">
            <p className="text-[10px] sm:text-xs text-amber-800 text-center leading-relaxed max-w-3xl mx-auto">
              <strong>Medical Disclaimer:</strong> This application is a
              machine-learning research and educational tool. It is <strong>not</strong> a medical
              diagnostic device. Predictions should not replace evaluation by a qualified healthcare
              professional. Always consult a physician for medical advice.
            </p>
          </div>

          {/* Footer Grid */}
          <div className="grid sm:grid-cols-3 gap-6 sm:gap-8 mb-6 sm:mb-8">
            {/* Developer Info */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Heart className="w-5 h-5 fill-red-500 text-red-500" />
                <span className="font-bold text-gray-900">HeartGuard ML</span>
              </div>
              <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                An explainable machine learning framework for heart disease risk prediction using public health data.
              </p>
              <div className="mt-3 space-y-1">
                <p className="text-sm font-semibold text-gray-800">Souvik Barui</p>
                <p className="text-xs text-gray-500">Research & Development</p>
                <a href="https://github.com/souvikbarui2003" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-xs text-primary-600 hover:underline mt-1">
                  <Github className="w-3 h-3" />github.com/souvikbarui2003
                </a>
                <a href="mailto:projectmakersb@gmail.com" className="inline-flex items-center gap-1 text-xs text-primary-600 hover:underline mt-1">
                  ✉️ projectmakersb@gmail.com
                </a>
              </div>
            </div>

            {/* Quick Links */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 text-sm">Quick Links</h3>
              <ul className="space-y-2 text-xs sm:text-sm">
                <li><button onClick={() => handleNav('predict')} className="text-gray-600 hover:text-primary-600 transition-colors">Risk Prediction</button></li>
                <li><button onClick={() => handleNav('dashboard')} className="text-gray-600 hover:text-primary-600 transition-colors">Model Dashboard</button></li>
                <li><button onClick={() => handleNav('research')} className="text-gray-600 hover:text-primary-600 transition-colors">Research Paper</button></li>
                <li><button onClick={() => handleNav('guidelines')} className="text-gray-600 hover:text-primary-600 transition-colors">Usage Guidelines</button></li>
                <li><a href="https://github.com/souvikbarui2003/ML-project-Heart-Disease" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-primary-600 transition-colors">GitHub Repository</a></li>
              </ul>
            </div>

            {/* Legal */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 text-sm">Legal</h3>
              <ul className="space-y-2 text-xs sm:text-sm">
                <li><button onClick={() => handleNav('terms')} className="text-gray-600 hover:text-primary-600 transition-colors">Terms & Conditions</button></li>
                <li><button onClick={() => handleNav('privacy')} className="text-gray-600 hover:text-primary-600 transition-colors">Privacy Policy</button></li>
                <li><button onClick={() => handleNav('guidelines')} className="text-gray-600 hover:text-primary-600 transition-colors">Usage Guidelines</button></li>
                <li><a href="/LICENSE" className="text-gray-600 hover:text-primary-600 transition-colors">MIT License</a></li>
              </ul>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="border-t border-gray-200 pt-4 sm:pt-6 flex flex-col sm:flex-row items-center justify-between gap-3">
            <p className="text-[10px] sm:text-xs text-gray-500 text-center sm:text-left">
              © 2024-2026 Souvik Barui. Built with ❤️ for healthcare AI research.
            </p>
            <p className="text-[10px] sm:text-xs text-gray-400 text-center sm:text-right">
              Heart Disease Risk Prediction System v2.0.0
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
