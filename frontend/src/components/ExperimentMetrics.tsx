import type {
  ExperimentContrastViewModel,
  ExperimentRunViewModel,
} from '../types/timeline'

export function metricText(value: unknown, digits = 3): string {
  const numeric = toMetric(value)
  return numeric === undefined ? '未计算' : numeric.toFixed(digits)
}

export function summaryMetricText(value: unknown, digits = 4, missing = '--'): string {
  const numeric = toMetric(value)
  return numeric === undefined ? missing : numeric.toFixed(digits)
}

function toMetric(value: unknown): number | undefined {
  if (value === null || value === undefined || value === '') {
    return undefined
  }
  const numeric = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numeric) ? numeric : undefined
}

function runDisplayName(run: { name?: string; runId?: string }): string {
  if (run.name && run.name !== '--') {
    return run.name
  }
  if (run.runId && run.runId !== '--') {
    return run.runId
  }
  return '--'
}

export function experimentTypeLabel(type?: string): string {
  const labels: Record<string, string> = {
    incremental_prediction: '增量预测',
    model_sensitivity: '模型敏感性',
    baseline_benchmark: '单臂基准',
    robustness_diagnostic: '稳健性诊断',
    temporal_sensitivity: '时间轴敏感性',
  }
  return type ? (labels[type] ?? type) : '未标注'
}

export function ContrastRows({ contrasts }: { contrasts: ExperimentContrastViewModel[] }) {
  if (!contrasts || contrasts.length === 0) {
    return null
  }
  return (
    <div className="comparison-list">
      {contrasts.map((contrast) => (
        <div key={contrast.contrastId} className="comparison-row">
          <span className="comparison-row__label">
            {contrast.name ?? contrast.contrastId}
            {contrast.note ? <em className="comparison-row__note"> · {contrast.note}</em> : null}
          </span>
          <div className="comparison-row__bars">
            <div className="comparison-bar comparison-bar--baseline">
              <span>
                Δ Pearson r {metricText(contrast.deltaPearsonR, 4)}
                {Array.isArray(contrast.pearsonCi) && contrast.pearsonCi.length === 2
                  ? ` · 95% CI ${metricText(contrast.pearsonCi[0], 4)} ~ ${metricText(contrast.pearsonCi[1], 4)}`
                  : ''}
              </span>
            </div>
            <div className="comparison-bar comparison-bar--treatment">
              <span>
                Δ RMSE {metricText(contrast.deltaRmse, 3)}
                {Array.isArray(contrast.rmseCi) && contrast.rmseCi.length === 2
                  ? ` · 95% CI ${metricText(contrast.rmseCi[0], 3)} ~ ${metricText(contrast.rmseCi[1], 3)}`
                  : ''}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export function RunMetricBox({ run }: { run: ExperimentRunViewModel }) {
  return (
    <div className="metric-box" key={run.runId}>
      <span className="metric-box__label">{runDisplayName(run)} · Pearson r</span>
      <span className="metric-box__value">{metricText(run.pearsonR, 4)}</span>
      <span className="metric-box__sub">
        RMSE {metricText(run.rmse, 3)} · MAE {metricText(run.mae, 3)}
      </span>
    </div>
  )
}

export function RunMetricsGrid({
  runs,
  experimentType,
}: {
  runs: ExperimentRunViewModel[]
  experimentType?: string
}) {
  if (!runs || runs.length === 0) {
    return <div className="timeline-empty-state">暂无运行指标。</div>
  }
  return (
    <div className="metrics-grid metrics-grid--two">
      {runs.map((run) => (
        <RunMetricBox key={run.runId} run={run} />
      ))}
      {experimentType === 'baseline_benchmark'
        ? (
            <div className="metric-box metric-box--note">
              <span className="metric-box__label">基准说明</span>
              <span className="metric-box__value">单臂实验</span>
              <span className="metric-box__sub">不构造对照组 / 实验组增益</span>
            </div>
          )
        : null}
    </div>
  )
}

export function DiagnosticsList({ diagnostics }: { diagnostics?: unknown[] }) {
  if (!diagnostics || diagnostics.length === 0) {
    return null
  }
  return (
    <div className="diagnostics-list">
      <span className="diagnostics-list__title">诊断结果（{diagnostics.length} 项）</span>
      <ul className="detail-list">
        {diagnostics.map((item, index) => (
          <li key={index}>
            {typeof item === 'string' ? item : JSON.stringify(item) ?? String(item)}
          </li>
        ))}
      </ul>
    </div>
  )
}

export function LegacyTwoArmGrid({
  baselinePearsonR,
  treatmentPearsonR,
  baselineRmse,
  treatmentRmse,
  deltaPearsonR,
  deltaRmse,
}: {
  baselinePearsonR?: number
  treatmentPearsonR?: number
  baselineRmse?: number
  treatmentRmse?: number
  deltaPearsonR?: number
  deltaRmse?: number
}) {
  return (
    <div className="metrics-grid metrics-grid--two">
      <div className="metric-box">
        <span className="metric-box__label">对照组 r</span>
        <span className="metric-box__value">{metricText(baselinePearsonR, 4)}</span>
      </div>
      <div className="metric-box">
        <span className="metric-box__label">实验组 r</span>
        <span className="metric-box__value">{metricText(treatmentPearsonR, 4)}</span>
      </div>
      <div className="metric-box metric-box--highlight">
        <span className="metric-box__label">Delta r</span>
        <span className="metric-box__value">{metricText(deltaPearsonR, 4)}</span>
      </div>
      <div className="metric-box">
        <span className="metric-box__label">Delta RMSE</span>
        <span className="metric-box__value">{metricText(deltaRmse, 3)}</span>
      </div>
      {baselineRmse !== undefined ? (
        <div className="metric-box">
          <span className="metric-box__label">对照组 RMSE</span>
          <span className="metric-box__value">{metricText(baselineRmse, 3)}</span>
        </div>
      ) : null}
      {treatmentRmse !== undefined ? (
        <div className="metric-box">
          <span className="metric-box__label">实验组 RMSE</span>
          <span className="metric-box__value">{metricText(treatmentRmse, 3)}</span>
        </div>
      ) : null}
    </div>
  )
}
