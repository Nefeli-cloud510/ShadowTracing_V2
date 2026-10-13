import { useState } from 'react'
import type { ApprovalOverlayViewModel } from '../types/timeline'

interface ApprovalOverlayProps {
  data: ApprovalOverlayViewModel
  open: boolean
  onClose: () => void
  onAction: (
    action: 'approve' | 'modify' | 'reject' | 'pause',
    payload: { candidateId?: string; humanNotes?: string },
  ) => Promise<void>
  busy?: boolean
  inline?: boolean
}

export function ApprovalOverlay({ data, open, onClose, onAction, busy = false, inline = false }: ApprovalOverlayProps) {
  const [selectedCandidate, setSelectedCandidate] = useState<string | undefined>(data.recommendation.candidateId)
  const [notes, setNotes] = useState('')

  if (!open && !inline) {
    return null
  }

  const content = (
    <div className={inline ? 'approval-inline-panel' : 'approval-overlay__panel'} onClick={(event) => !inline && event.stopPropagation()}>
        <div className="approval-overlay__header">
          <div>
            <span className="detail-card__eyebrow">Approval Request</span>
            <h2>审批请求 · Round {data.roundNumber}</h2>
          </div>
          {inline ? null : (
            <button
              type="button"
              className="detail-link detail-link--button"
              onClick={onClose}
            >
              关闭
            </button>
          )}
        </div>

        <div className="approval-overlay__table">
          <div className="approval-overlay__row approval-overlay__row--head">
            <span>实验</span>
            <span>预期区分度</span>
            <span>性能增益</span>
            <span>风险</span>
            <span>成本</span>
            <span>综合价值</span>
          </div>
          {data.candidates.map((item) => (
            <div key={item.experimentId}>
              <div
                className={[
                  'approval-overlay__row',
                  item.recommended ? 'approval-overlay__row--recommended' : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
              >
                <span>{item.recommended ? `★${item.experimentId}` : item.experimentId}</span>
                <span>{item.informationGain.toFixed(3)}</span>
                <span>{item.performanceGain.toFixed(3)}</span>
                <span>{item.risk.toFixed(3)}</span>
                <span>{item.cost.toFixed(3)}</span>
                <span>{item.utility.toFixed(3)}</span>
              </div>
              <div className="approval-overlay__summary">
                <strong>{item.experimentMode === 'baseline' ? '对照组实验（单组）' : '区分性对照实验'}</strong>
                <p>对照组变量：{item.experimentMode === 'baseline' ? item.treatmentVariables.join('、') || '未配置' : item.controlVariables.join('、') || '未配置'}</p>
                <p>实验组变量：{item.experimentMode === 'baseline' ? '不设置实验组' : item.treatmentVariables.join('、') || '未配置'}</p>
                <p>{item.hasFeatureDifference ? '校验结果：两组变量存在差异，可开展对照。' : '校验结果：两组变量无差异，当前方案应被拦截。'}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="approval-overlay__summary">
          <strong>推荐理由</strong>
          <p>{data.recommendation.reason}</p>
        </div>

        <div className="controller-input">
          <label className="controller-input__field">
            <span>目标候选实验</span>
            <input
              className="controller-inline-input"
              value={selectedCandidate ?? ''}
              onChange={(event) => setSelectedCandidate(event.target.value)}
              placeholder="默认使用推荐实验 ID"
            />
          </label>
          <label className="controller-input__field">
            <span>人工备注 / 参数调整说明</span>
            <textarea
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              placeholder="输入批准理由、参数修改建议、拒绝原因或暂停说明。"
              rows={4}
            />
          </label>
        </div>

        <div className="approval-overlay__actions">
          <button
            type="button"
            className="detail-link detail-link--button"
            onClick={() => void onAction('approve', { candidateId: selectedCandidate, humanNotes: notes })}
            disabled={busy}
          >
            {busy ? '加载中…' : '批准并继续'}
          </button>
          <button
            type="button"
            className="detail-link detail-link--button"
            onClick={() => void onAction('modify', { candidateId: selectedCandidate, humanNotes: notes })}
            disabled={busy}
          >
            修改参数重选
          </button>
          <button
            type="button"
            className="detail-link detail-link--button"
            onClick={() => void onAction('reject', { candidateId: selectedCandidate, humanNotes: notes })}
            disabled={busy}
          >
            拒绝并重选
          </button>
          <button
            type="button"
            className="detail-link detail-link--button"
            onClick={() => void onAction('pause', { candidateId: selectedCandidate, humanNotes: notes })}
            disabled={busy}
          >
            暂停
          </button>
        </div>
      </div>
  )

  if (inline) {
    return content
  }

  return (
    <div className="approval-overlay" role="dialog" aria-modal="true" onClick={onClose}>
      {content}
    </div>
  )
}
