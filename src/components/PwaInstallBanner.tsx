import { useState, useEffect } from 'react'
import { Download, X } from 'lucide-react'

export default function PwaInstallBanner() {
  const [show, setShow] = useState(false)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    // Check if already dismissed or installed
    const isDismissed = localStorage.getItem('pwa-install-dismissed')
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches

    if (isDismissed || isStandalone) return

    const handler = () => setShow(true)
    window.addEventListener('pwa-install-available', handler)

    // Show after 3 seconds if prompt is available
    const timeout = setTimeout(() => {
      if (window.__installPrompt) setShow(true)
    }, 3000)

    return () => {
      window.removeEventListener('pwa-install-available', handler)
      clearTimeout(timeout)
    }
  }, [])

  const handleInstall = async () => {
    if (!window.__installPrompt) return
    window.__installPrompt.prompt()
    const { outcome } = await window.__installPrompt.userChoice
    if (outcome === 'accepted') {
      setShow(false)
    }
    window.__installPrompt = null
  }

  const handleDismiss = () => {
    setShow(false)
    setDismissed(true)
    localStorage.setItem('pwa-install-dismissed', 'true')
  }

  if (!show || dismissed) return null

  return (
    <div className="fixed bottom-4 left-4 right-4 sm:left-auto sm:right-4 sm:max-w-sm z-50 animate-slide-up">
      <div className="bg-white rounded-2xl shadow-2xl border border-gray-200 p-4 flex items-start gap-3">
        <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center shrink-0">
          <Download className="w-5 h-5 text-primary-600" />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="font-semibold text-gray-900 text-sm">Install HeartGuard ML</h4>
          <p className="text-xs text-gray-500 mt-0.5">Add to your home screen for quick access</p>
          <div className="flex items-center gap-2 mt-2">
            <button
              onClick={handleInstall}
              className="bg-primary-600 text-white text-xs font-semibold px-4 py-1.5 rounded-lg hover:bg-primary-700 transition-colors"
            >
              Install
            </button>
            <button
              onClick={handleDismiss}
              className="text-gray-400 text-xs hover:text-gray-600 transition-colors"
            >
              Not now
            </button>
          </div>
        </div>
        <button
          onClick={handleDismiss}
          className="text-gray-400 hover:text-gray-600 transition-colors shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

// Extend Window interface for TypeScript
declare global {
  interface Window {
    __installPrompt: any
  }
}
