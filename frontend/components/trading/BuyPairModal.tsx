// frontend/components/trading/BuyPairModal.tsx
'use client'

import { useState } from 'react'
import { useWallet } from '@/hooks/useWallet'
import { getApiBaseUrl } from '@/config/api-config'
import type { Portfolio } from '@/types/portfolio'

interface BuyPairModalProps {
  portfolio: Portfolio
  onClose: () => void
}

type Step = 'input' | 'confirming' | 'executing' | 'success' | 'error'

interface TradeResult {
  success: boolean
  target: { split_tx?: string; clob_order_id?: string; error?: string }
  cover: { split_tx?: string; clob_order_id?: string; error?: string }
  total_spent: number
  final_balances: { pol: number; usdc_e: number }
  warnings?: string[]
}

export function BuyPairModal({ portfolio: p, onClose }: BuyPairModalProps) {
  const { status } = useWallet()
  const [amount, setAmount] = useState('10')
  const [step, setStep] = useState<Step>('input')
  const [result, setResult] = useState<TradeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [executionStep, setExecutionStep] = useState('')

  const apiBase = getApiBaseUrl()
  const amountNum = parseFloat(amount) || 0
  const totalCost = amountNum * 2
  const hasSufficientBalance = (status?.balances?.usdc_e || 0) >= totalCost

  const handleBuy = async () => {
    if (!status?.unlocked) {
      setError('Please unlock your wallet first')
      return
    }
    if (!hasSufficientBalance) {
      setError('Insufficient USDC.e balance')
      return
    }

    setStep('executing')
    setError(null)
    setExecutionStep('Splitting target position...')

    try {
      const res = await fetch(`${apiBase}/trading/buy-pair`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pair_id: p.pair_id,
          target_market_id: p.target_market_id,
          target_position: p.target_position,
          cover_market_id: p.cover_market_id,
          cover_position: p.cover_position,
          amount_per_position: amountNum,
          skip_clob_sell: false,
        }),
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || 'Trade failed')
      }

      setResult(data)
      setStep(data.success ? 'success' : 'error')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Trade failed')
      setStep('error')
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-void/80 backdrop-blur-sm" />

      <div
        className="relative w-full max-w-md bg-surface border border-border rounded-xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="h-0.5 bg-cyan" />
        <div className="px-5 py-4 border-b border-border flex items-center justify-between">
          <h2 className="text-lg font-semibold text-text-primary">Buy Pair On-Chain</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          {step === 'input' && (
            <>
              {/* Positions */}
              <div className="space-y-2">
                <div className="bg-surface-elevated rounded-lg p-3">
                  <div className="text-[10px] text-text-muted uppercase tracking-wide mb-1">Target</div>
                  <p className="text-sm text-text-primary">{p.target_question.slice(0, 50)}...</p>
                  <span className={`text-xs font-mono ${p.target_position === 'YES' ? 'text-emerald' : 'text-rose'}`}>
                    {p.target_position} @ ${p.target_price.toFixed(2)}
                  </span>
                </div>
                <div className="bg-surface-elevated rounded-lg p-3">
                  <div className="text-[10px] text-text-muted uppercase tracking-wide mb-1">Cover</div>
                  <p className="text-sm text-text-primary">{p.cover_question.slice(0, 50)}...</p>
                  <span className={`text-xs font-mono ${p.cover_position === 'YES' ? 'text-emerald' : 'text-rose'}`}>
                    {p.cover_position} @ ${p.cover_price.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Amount input */}
              <div>
                <label className="text-sm text-text-muted block mb-1">Amount per position</label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted">$</span>
                  <input
                    type="number"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    min="1"
                    step="1"
                    className="w-full pl-7 pr-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary text-sm font-mono focus:outline-none focus:border-cyan"
                  />
                </div>
              </div>

              {/* Summary */}
              <div className="bg-surface-elevated rounded-lg p-3 space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">2 positions × ${amountNum.toFixed(2)}</span>
                  <span className="text-text-primary font-mono">${totalCost.toFixed(2)} total</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Your balance</span>
                  <span className={`font-mono ${hasSufficientBalance ? 'text-emerald' : 'text-rose'}`}>
                    ${(status?.balances?.usdc_e || 0).toFixed(2)} USDC.e
                  </span>
                </div>
              </div>

              {error && (
                <div className="p-3 bg-rose/10 border border-rose/25 rounded-lg text-rose text-sm">
                  {error}
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={onClose}
                  className="flex-1 py-2.5 px-4 border border-border rounded-lg text-text-muted hover:text-text-primary text-sm transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleBuy}
                  disabled={!status?.unlocked || !hasSufficientBalance || amountNum <= 0}
                  className="flex-1 py-2.5 px-4 bg-cyan hover:bg-cyan/90 rounded-lg text-void text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {!status?.unlocked ? 'Unlock Wallet First' : 'Confirm Purchase'}
                </button>
              </div>
            </>
          )}

          {step === 'executing' && (
            <div className="py-8 text-center">
              <div className="animate-spin w-8 h-8 border-2 border-cyan border-t-transparent rounded-full mx-auto mb-4" />
              <p className="text-text-primary">{executionStep}</p>
              <p className="text-text-muted text-sm mt-2">This may take a minute...</p>
            </div>
          )}

          {step === 'success' && result && (
            <div className="py-4 space-y-4">
              <div className="text-center">
                <div className={`w-12 h-12 ${result.warnings?.length ? 'bg-amber-500/20' : 'bg-emerald/20'} rounded-full flex items-center justify-center mx-auto mb-3`}>
                  {result.warnings?.length ? (
                    <svg className="w-6 h-6 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  ) : (
                    <svg className="w-6 h-6 text-emerald" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
                <h3 className="text-lg font-semibold text-text-primary">
                  {result.warnings?.length ? 'Partial Success' : 'Purchase Complete'}
                </h3>
              </div>

              {/* Warnings */}
              {result.warnings && result.warnings.length > 0 && (
                <div className="bg-amber-500/10 border border-amber-500/25 rounded-lg p-3 space-y-1">
                  {result.warnings.map((warning, i) => (
                    <p key={i} className="text-amber-500 text-sm">{warning}</p>
                  ))}
                </div>
              )}

              <div className="bg-surface-elevated rounded-lg p-3 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">Total spent</span>
                  <span className="text-text-primary font-mono">${result.total_spent.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">New balance</span>
                  <span className="text-text-primary font-mono">${result.final_balances.usdc_e.toFixed(2)}</span>
                </div>
              </div>

              {result.target.split_tx && (
                <div className="text-xs text-text-muted">
                  Target TX: <code className="text-cyan">{result.target.split_tx.slice(0, 20)}...</code>
                </div>
              )}
              {result.cover.split_tx && (
                <div className="text-xs text-text-muted">
                  Cover TX: <code className="text-cyan">{result.cover.split_tx.slice(0, 20)}...</code>
                </div>
              )}

              <button
                onClick={onClose}
                className="w-full py-2.5 px-4 bg-surface-elevated hover:bg-surface border border-border rounded-lg text-text-primary text-sm transition-colors"
              >
                Close
              </button>
            </div>
          )}

          {step === 'error' && (
            <div className="py-4 space-y-4">
              <div className="text-center">
                <div className="w-12 h-12 bg-rose/20 rounded-full flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-rose" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-text-primary">Trade Failed</h3>
                <p className="text-rose text-sm mt-2">{error}</p>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={onClose}
                  className="flex-1 py-2.5 px-4 border border-border rounded-lg text-text-muted text-sm"
                >
                  Close
                </button>
                <button
                  onClick={() => { setStep('input'); setError(null); }}
                  className="flex-1 py-2.5 px-4 bg-cyan hover:bg-cyan/90 rounded-lg text-void text-sm font-medium"
                >
                  Try Again
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
