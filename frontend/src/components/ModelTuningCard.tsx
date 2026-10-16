import type { ExperimentEvaluationViewModel } from '../types/timeline'

type TuningEntry = NonNullable<ExperimentEvaluationViewModel['tuningEntries']>[number]

const PARAMETER_LABELS: Record<string, string> = {
  window_size: '窗口大小（天）',
  past_lag_days: '样本滞后（天）',
  forecast_horizon_days: '预测视野（天）',
  max_lag_day: '最大滞后构造（天）',
  test_split_ratio: '迭代评估比例',
  alpha: '正则强度 alpha',
  l1_ratio: 'L1 占比 l1_ratio',
  use_lag_feature: '是否启用滞后特征',
  random_state: '随机种子',
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '--'
  if (typeof value === 'boolean') return value ? '启用' : '不启用'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function ParameterList({ parameters }: { parameters: Record<string, unknown> }) {
  const entries = Object.entries(parameters).filter(
    ([, value]) => value === null || typeof value !== 'object',
  )
  if (entries.length === 0) {
    return null
  }
  return (
    <ul className="model-parameter-list">
      {entries.map(([key, value]) => (
        <li key={key}>
          <strong>{PARAMETER_LABELS[key] ?? key}</strong>
          <span>{formatValue(value)}</span>
        </li>
      ))}
    </ul>
  )
}

export function ModelTuningCard({
  modelParameters,
  tuningEntries,
  tuningNarrative,
}: {
  modelParameters?: Record<string, unknown>
  tuningEntries?: TuningEntry[]
  tuningNarrative?: string
}) {
  const hasTuning = Boolean(tuningEntries && tuningEntries.length > 0)
  const hasFinalParameters = Boolean(modelParameters && Object.keys(modelParameters).length > 0)
  return (
    <section className="detail-card detail-card--wide model-tuning-card">
      <span className="detail-card__eyebrow">Model Parameters</span>
      <h2>本轮模型参数与调参理由</h2>
      {tuningNarrative ? (
        <div className="evaluation-summary-card">
          <p>{tuningNarrative}</p>
        </div>
      ) : null}
      {hasTuning ? (
        <div className="collection-grid">
          {tuningEntries!.map((entry, index) => (
            <article key={`${entry.refinementType ?? 'tuning'}-${index}`} className="collection-card">
              <strong>{entry.refinementType ?? '模型调参'}</strong>
              {!tuningNarrative ? <p>{entry.rationale || '未记录调整理由。'}</p> : null}
              {entry.modelParameters && Object.keys(entry.modelParameters).length > 0 ? (
                <ParameterList parameters={entry.modelParameters} />
              ) : (
                <div className="collection-card__meta">
                  <span>无额外参数变更</span>
                </div>
              )}
              {!tuningNarrative && entry.protocolNotes && entry.protocolNotes.length > 0 ? (
                <ul className="detail-list">
                  {entry.protocolNotes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              ) : null}
            </article>
          ))}
        </div>
      ) : null}
      {hasFinalParameters ? (
        <div className="evaluation-summary-card">
          <h3>最终执行参数</h3>
          <ParameterList parameters={modelParameters!} />
        </div>
      ) : (
        <p className="detail-card__meta">本轮无显式模型参数调整，沿用协议默认参数。</p>
      )}
    </section>
  )
}
