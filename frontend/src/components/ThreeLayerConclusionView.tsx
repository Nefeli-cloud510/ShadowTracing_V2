import type {
  ExperimentContrastViewModel,
  ModelSensitivityFamilyEvidenceViewModel,
  ThreeLayerConclusionExperimentLayer,
  ThreeLayerConclusionViewModel,
} from '../types/timeline'
import { getVisualizationPreviewUrl } from '../data/realStateLoader'
import { ContrastRows, RunMetricsGrid, experimentTypeLabel } from './ExperimentMetrics'

function formatNumber(value?: number, digits = 4): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return '--'
  }
  return value.toFixed(digits)
}

function formatRange(range?: [number, number]): string {
  if (!range || range.length < 2) {
    return '--'
  }
  return `${Number(range[0]).toFixed(4)} ~ ${Number(range[1]).toFixed(4)}`
}

function directionLabel(value?: string): string {
  const labels: Record<string, string> = {
    positive: '正增益',
    negative: '负增益',
    zero: '零',
    near_zero: '接近零',
    mixed: '混合',
    unknown: '未定',
  }
  return value ? (labels[value] ?? value) : '--'
}

function matchLabel(value?: boolean | string): string {
  if (value === true) {
    return '匹配'
  }
  if (value === false) {
    return '偏离'
  }
  const text = typeof value === 'string' ? value.toLowerCase() : ''
  if (text === 'matched' || text === 'true') {
    return '匹配'
  }
  if (text === 'mismatched' || text === 'unmatched' || text === 'false') {
    return '偏离'
  }
  return '--'
}

function chartRoleLabel(role: string): string {
  const exactLabels: Record<string, string> = {
    baseline_timeseries: '对照组时序',
    treatment_timeseries: '实验组时序',
    baseline_scatter: '对照组散点',
    treatment_scatter: '实验组散点',
  }
  if (exactLabels[role]) {
    return exactLabels[role]
  }
  const model = ['ridge', 'elasticnet', 'lasso', 'linear'].find((token) =>
    role.toLowerCase().includes(token),
  )
  const arm = role.toLowerCase().includes('treatment')
    ? '实验组'
    : role.toLowerCase().includes('baseline') || role.toLowerCase().includes('control')
      ? '对照组'
      : ''
  const chart = role.toLowerCase().includes('scatter')
    ? '散点'
    : role.toLowerCase().includes('timeseries') || role.toLowerCase().includes('series')
      ? '时序'
      : ''
  return [model, arm, chart].filter(Boolean).join(' ') || role || '未命名图表'
}

function classForMatch(value?: boolean | string): string {
  if (value === true) {
    return 'match'
  }
  if (value === false) {
    return 'miss'
  }
  const text = typeof value === 'string' ? value.toLowerCase() : ''
  if (text === 'matched' || text === 'true' || text === 'partial') {
    return 'match'
  }
  if (text === 'mismatched' || text === 'unmatched' || text === 'false') {
    return 'miss'
  }
  return ''
}

function familyEvaluationLabel(evaluation: string): string {
  const labels: Record<string, string> = {
    stable_positive: '稳定正向增益',
    stable_negative: '稳定负向增益',
    no_stable_gain: '未检出稳定增益（CI 跨零）',
    insufficient_blocks: 'CI 不可用 / 有效时间块不足',
    point_positive: '点估计正向，无稳定证据',
    point_negative: '点估计负向，无稳定证据',
    point_zero: '点估计接近零',
    not_evaluated: '未评估',
  }
  return labels[evaluation] ?? evaluation
}

function classifyFamilyGain(point?: number, rmseCi?: number[] | null): string {
  const ci = Array.isArray(rmseCi) && rmseCi.length >= 2 ? rmseCi : undefined
  const lower = ci ? Number(ci[0]) : Number.NaN
  const upper = ci ? Number(ci[1]) : Number.NaN
  if (ci && Number.isFinite(lower) && Number.isFinite(upper)) {
    if ((lower < 0 && upper > 0) || lower === 0 || upper === 0) {
      return 'no_stable_gain'
    }
    if (lower > 0) {
      return 'stable_positive'
    }
    if (upper < 0) {
      return 'stable_negative'
    }
    return 'insufficient_blocks'
  }
  if (point === undefined || point === null || !Number.isFinite(point)) {
    return 'not_evaluated'
  }
  return point > 0 ? 'point_positive' : point < 0 ? 'point_negative' : 'point_zero'
}

function deriveFamilySensitivityRows(
  experiment: ThreeLayerConclusionExperimentLayer,
): ModelSensitivityFamilyEvidenceViewModel[] {
  const runs = experiment.runs ?? []
  const runById = new Map(runs.map((run) => [run.runId, run]))
  const byFamily = new Map<string, ExperimentContrastViewModel>()
  for (const contrast of experiment.contrasts ?? []) {
    let family = 'unknown'
    for (const runId of contrast.runIds ?? []) {
      const run = runById.get(runId)
      if (run?.modelFamily) {
        family = run.modelFamily
        break
      }
    }
    if (!byFamily.has(family)) {
      byFamily.set(family, contrast)
    }
  }
  return Array.from(byFamily.entries()).map(([family, contrast]) => {
    const evaluation = classifyFamilyGain(contrast.deltaRmse, contrast.rmseCi)
    return {
      family,
      contrastId: contrast.contrastId,
      pointGainRmse: contrast.deltaRmse,
      pointGainPearsonR: contrast.deltaPearsonR,
      rmseCi: contrast.rmseCi,
      pearsonCi: contrast.pearsonCi,
      crossesZero:
        Array.isArray(contrast.rmseCi) && contrast.rmseCi.length >= 2
          ? contrast.rmseCi[0] <= 0 && contrast.rmseCi[1] >= 0
          : null,
      evaluation,
      evaluationLabel: familyEvaluationLabel(evaluation),
    }
  })
}

function sensitivityOverallLabel(
  rows: ModelSensitivityFamilyEvidenceViewModel[],
): string {
  if (!rows || rows.length === 0) {
    return '未评估'
  }
  const statuses = rows.map((row) => row.evaluation)
  if (
    statuses.some((status) =>
      ['no_stable_gain', 'insufficient_blocks', 'not_evaluated'].includes(status),
    )
  ) {
    return '未检出稳定增益'
  }
  const stablePositive = statuses.filter((status) => status === 'stable_positive').length
  const stableNegative = statuses.filter((status) => status === 'stable_negative').length
  if (stablePositive > 0 && stableNegative > 0) {
    return '混合方向（需先核查数据与执行）'
  }
  if (stablePositive === rows.length) {
    return '稳定正向增益'
  }
  if (stableNegative === rows.length) {
    return '稳定负向增益'
  }
  return '未检出稳定增益'
}

function formatCi(ci?: number[] | null): string {
  if (!Array.isArray(ci) || ci.length < 2 || !ci.every((value) => Number.isFinite(value))) {
    return '--'
  }
  return `${Number(ci[0]).toFixed(4)} ~ ${Number(ci[1]).toFixed(4)}`
}

function hypothesisGroup(
  row: ThreeLayerConclusionViewModel['hypothesisLayer'][number],
): 'supported' | 'partial' | 'weakened' | 'untested' {
  const direction = row.directionMatched
  const magnitude = row.magnitudeMatched
  if (direction === 'partial' || magnitude === 'partial') {
    return 'partial'
  }
  if (direction === false || direction === 'mismatched' || direction === 'unmatched') {
    return 'weakened'
  }
  if (direction === true || direction === 'matched' || direction === 'true') {
    return magnitude === false || magnitude === 'mismatched' || magnitude === 'unmatched'
      ? 'partial'
      : 'supported'
  }
  if (/削弱|不符|不支持/i.test(row.conclusion)) {
    return 'weakened'
  }
  if (/部分|幅度不足/i.test(row.conclusion)) {
    return 'partial'
  }
  if (/获得支持|一致/i.test(row.conclusion)) {
    return 'supported'
  }
  return 'untested'
}

const GROUP_META: Record<
  'supported' | 'partial' | 'weakened' | 'untested',
  { label: string; className: string }
> = {
  supported: { label: '获得支持', className: 'supported' },
  partial: { label: '部分支持', className: 'partial' },
  weakened: { label: '被削弱', className: 'weakened' },
  untested: { label: '未直接检验', className: 'untested' },
}

export function ThreeLayerConclusionView({
  conclusion,
  roundNumber,
}: {
  conclusion: ThreeLayerConclusionViewModel
  roundNumber?: number
}) {
  const experiment = conclusion.experimentLayer
  const scientific = conclusion.scientificLayer
  const dataLayer = conclusion.dataLayer
  const trackingLayer = conclusion.trackingLayer
  const chartAnalyses = dataLayer?.chartAnalyses ?? []
  const familyRows =
    experiment.modelSensitivityFamilies && experiment.modelSensitivityFamilies.length > 0
      ? experiment.modelSensitivityFamilies
      : deriveFamilySensitivityRows(experiment)
  const familyOverallLabel = sensitivityOverallLabel(familyRows)
  const groupedRows = conclusion.hypothesisLayer.reduce<
    Record<'supported' | 'partial' | 'weakened' | 'untested', typeof conclusion.hypothesisLayer>
  >(
    (acc, row) => {
      acc[hypothesisGroup(row)].push(row)
      return acc
    },
    { supported: [], partial: [], weakened: [], untested: [] },
  )

  return (
    <section className="detail-card detail-card--wide three-layer-conclusion">
      <span className="detail-card__eyebrow">Round {roundNumber ?? 1} Result</span>
      <h2>第{roundNumber ?? 1}轮实验结果报告</h2>

      <div className="four-layer-conclusion">
        <div className="four-layer-conclusion__main">
          <div className="three-layer-conclusion__block">
            <h3>
              指标快照 · 本轮实验
              {experiment.experimentType ? ` · ${experimentTypeLabel(experiment.experimentType)}` : ''}
            </h3>
            <div className="three-layer-conclusion__experiment">
              <div className="three-layer-conclusion__experiment-main">
                <div className="three-layer-conclusion__experiment-id">
                  <span>本轮实验</span>
                  <strong>{experiment.experimentId || '--'}</strong>
                </div>
                <p>{experiment.designSummary || '尚未形成实验设计摘要。'}</p>
              </div>
              {experiment.runs && experiment.runs.length > 0 ? (
                <>
                  <div className="three-layer-conclusion__dynamic-metrics">
                    <RunMetricsGrid
                      runs={experiment.runs}
                      experimentType={experiment.experimentType}
                    />
                    <div
                      className={
                        experiment.decisive
                          ? 'three-layer-conclusion__metric three-layer-conclusion__metric--decisive'
                          : 'three-layer-conclusion__metric'
                      }
                    >
                      <span>判定</span>
                      <strong>{experiment.decisive ? '假设可区分' : '无法区分假设'}</strong>
                    </div>
                  </div>
                  <ContrastRows contrasts={experiment.contrasts ?? []} />
                  {experiment.experimentType === 'model_sensitivity' && familyRows.length > 0 ? (
                    <div className="model-sensitivity-families">
                      <div className="model-sensitivity-families__head">
                        <strong>各模型家族日影增量与 CI</strong>
                        <span>总体判定：{familyOverallLabel}</span>
                      </div>
                      <div className="model-sensitivity-families__table-wrap">
                        <table className="model-sensitivity-families__table">
                          <thead>
                            <tr>
                              <th>模型家族</th>
                              <th>ΔRMSE</th>
                              <th>95% CI（RMSE）</th>
                              <th>ΔPearson r</th>
                              <th>95% CI（Pearson r）</th>
                              <th>判定</th>
                            </tr>
                          </thead>
                          <tbody>
                            {familyRows.map((row) => (
                              <tr key={`${row.family}-${row.contrastId ?? 'contrast'}`}>
                                <td>{row.family}</td>
                                <td>{formatNumber(row.pointGainRmse, 4)}</td>
                                <td>{formatCi(row.rmseCi)}</td>
                                <td>{formatNumber(row.pointGainPearsonR, 4)}</td>
                                <td>{formatCi(row.pearsonCi)}</td>
                                <td>{row.evaluationLabel ?? familyEvaluationLabel(row.evaluation)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="model-sensitivity-families__note">
                        点估计方向与稳定证据分开判断；任一家族 CI 跨零或有效时间块不足时，
                        整体结论固定为“未检出稳定增益”，本轮不自动改变假设支持度。
                      </p>
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="three-layer-conclusion__metrics">
                  <div className="three-layer-conclusion__metric">
                    <span>对照组 RMSE</span>
                    <strong>{formatNumber(experiment.baselineRmse)}</strong>
                  </div>
                  <div className="three-layer-conclusion__metric">
                    <span>实验组 RMSE</span>
                    <strong>{formatNumber(experiment.treatmentRmse)}</strong>
                  </div>
                  <div className="three-layer-conclusion__metric three-layer-conclusion__metric--accent">
                    <span>对照组 Pearson-r</span>
                    <strong>{formatNumber(experiment.baselinePearsonR)}</strong>
                  </div>
                  <div className="three-layer-conclusion__metric">
                    <span>实验组 Pearson-r</span>
                    <strong>{formatNumber(experiment.treatmentPearsonR)}</strong>
                  </div>
                  <div
                    className={
                      experiment.decisive
                        ? 'three-layer-conclusion__metric three-layer-conclusion__metric--decisive'
                        : 'three-layer-conclusion__metric'
                    }
                  >
                    <span>判定</span>
                    <strong>{experiment.decisive ? '假设可区分' : '无法区分假设'}</strong>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="three-layer-conclusion__block">
            <h3>第一层 · 数据层：数值分析与归因</h3>
              <div className="three-layer-conclusion__data">
              <div className="three-layer-conclusion__data-item">
                <span>RMSE 变化归因</span>
                <p>{dataLayer?.rmseAttribution ?? '本轮暂未生成 RMSE 数值归因，由程序指标回退展示。'}</p>
              </div>
              <div className="three-layer-conclusion__data-item">
                <span>Pearson-r 变化归因</span>
                <p>{dataLayer?.pearsonAttribution ?? '本轮暂未生成 Pearson-r 数值归因，由程序指标回退展示。'}</p>
              </div>
              <div className="three-layer-conclusion__data-item">
                <span>异常 / 不一致识别</span>
                <ul className="detail-list">
                  {dataLayer?.anomalies && dataLayer.anomalies.length > 0
                    ? dataLayer.anomalies.map((item) => <li key={item}>{item}</li>)
                    : <li>本轮未发现明显指标冲突，仍须结合稳健性分析确认。</li>}
                </ul>
              </div>
              <div className="three-layer-conclusion__data-item">
                <span>推荐后续关注方向</span>
                <p>{dataLayer?.nextFocus ?? '下一轮先围绕数据覆盖与路径控制实验继续收敛。'}</p>
              </div>
              {dataLayer?.comparisonAnalysis ? (
                <div className="three-layer-conclusion__data-item three-layer-conclusion__data-item--full">
                  <span>两组实验对比分析</span>
                  <p>{dataLayer.comparisonAnalysis}</p>
                </div>
              ) : null}
            </div>
            {chartAnalyses.length > 0 ? (
              <div className="three-layer-conclusion__charts">
                <span className="three-layer-conclusion__charts-title">本轮图表分析（真实图像）</span>
                <div className="three-layer-conclusion__chart-grid">
                  {chartAnalyses.map((chart) => (
                    <article className="three-layer-conclusion__chart" key={chart.chartRole}>
                      <div className="three-layer-conclusion__chart-head">
                        <strong>{chart.chartName || chartRoleLabel(chart.chartRole)}</strong>
                        <span>{chartRoleLabel(chart.chartRole)}</span>
                      </div>
                      {chart.chartImagePath ? (
                        <img
                          className="three-layer-conclusion__chart-image"
                          src={getVisualizationPreviewUrl(chart.chartImagePath)}
                          alt={chart.chartName || chartRoleLabel(chart.chartRole)}
                        />
                      ) : null}
                      <p>{chart.description}</p>
                      {chart.keyObservations.length > 0 ? (
                        <ul className="detail-list">
                          {chart.keyObservations.map((obs) => (
                            <li key={obs}>{obs}</li>
                          ))}
                        </ul>
                      ) : null}
                      {chart.anomalyOrInsight ? (
                        <p className="three-layer-conclusion__chart-insight">
                          洞察 / 异常：{chart.anomalyOrInsight}
                        </p>
                      ) : null}
                    </article>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div className="three-layer-conclusion__block">
            <h3>第二层 · 假设层：支持度判定</h3>
            <div className="three-layer-conclusion__summary">
              {(Object.keys(GROUP_META) as Array<keyof typeof groupedRows>).map((group) => {
                const meta = GROUP_META[group]
                const rows = groupedRows[group]
                return (
                  <div key={group} className={`three-layer-conclusion__summary-item ${meta.className}`}>
                    <strong>{meta.label}</strong>
                    <span>{rows.length > 0 ? rows.map((row) => row.displayHypothesisId).join('、') : '暂无'}</span>
                  </div>
                )
              })}
            </div>
            {conclusion.hypothesisLayerSummary ? (
              <div className="three-layer-conclusion__llm-summary">
                <span>假设层综合分析（科学解释者 LLM）</span>
                <p>{conclusion.hypothesisLayerSummary}</p>
              </div>
            ) : null}
            {conclusion.hypothesisLayer.length > 0 ? (
              <div className="three-layer-conclusion__table-wrap">
                <table className="three-layer-conclusion__table">
                  <thead>
                    <tr>
                      <th>假设</th>
                      <th>陈述</th>
                      <th>预测方向</th>
                      <th>预测范围</th>
                      <th>实际增量</th>
                      <th>方向匹配</th>
                      <th>幅度匹配</th>
                      <th>假设层结论</th>
                      <th>支持度</th>
                    </tr>
                  </thead>
                  <tbody>
                    {conclusion.hypothesisLayer.map((row) => (
                      <tr key={row.hypothesisId || row.displayHypothesisId}>
                        <td>
                          <span className="three-layer-conclusion__id">{row.displayHypothesisId}</span>
                        </td>
                        <td>{row.statement}</td>
                        <td>{directionLabel(row.predictedDirection)}</td>
                        <td>{formatRange(row.predictedRange)}</td>
                        <td>{formatNumber(row.actualDelta)}</td>
                        <td>
                          <span className={`three-layer-conclusion__match ${classForMatch(row.directionMatched)}`}>
                            {matchLabel(row.directionMatched)}
                          </span>
                        </td>
                        <td>
                          <span className={`three-layer-conclusion__match ${classForMatch(row.magnitudeMatched)}`}>
                            {matchLabel(row.magnitudeMatched)}
                          </span>
                        </td>
                        <td>{row.conclusion}</td>
                        <td>{formatNumber(row.supportAfter, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="detail-card__meta">本轮尚未形成假设判定表。</p>
            )}
            <div className="three-layer-conclusion__rules">
              <span>状态更新规则（程序执行）</span>
              <p>支持度上升 → 维持活跃或转收敛；支持度下降且低于 0.40 → 转观察；低于 0.20 → 剪枝；父假设剪枝 → 子假设继承剪枝。</p>
            </div>
          </div>

          <div className="three-layer-conclusion__block">
            <h3>第三层 · 科学问题层：对主科学问题的回答</h3>
            <div className="three-layer-conclusion__scientific">
              <div className="three-layer-conclusion__question">
                <span>主科学问题</span>
                <strong>{scientific.mainQuestion}</strong>
                <p>{scientific.answer || '本轮尚未形成明确科学结论。'}</p>
              </div>
              {scientific.pathQuestion || scientific.pathAnswer ? (
                <div className="three-layer-conclusion__question">
                  <span>传递路径</span>
                  <strong>{scientific.pathQuestion ?? '物理传递路径'}</strong>
                  <p>{scientific.pathAnswer ?? '本轮未形成路径结论。'}</p>
                </div>
              ) : null}
              {scientific.evidenceText ? (
                <p className="three-layer-conclusion__evidence">{scientific.evidenceText}</p>
              ) : null}
            </div>
          </div>

          <div className="three-layer-conclusion__block">
            <h3>第四层 · 实验追踪层：审计与存证</h3>
            <div className="three-layer-conclusion__tracking">
              <div>
                <span>审计项</span>
                <ul className="detail-list">
                  {(trackingLayer?.auditItems ?? ['本轮尚未写入审计项。']).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <span>来源模块</span>
                <ul className="detail-list">
                  {(trackingLayer?.sources ?? ['--']).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <span>快照引用</span>
                <ul className="detail-list">
                  {(trackingLayer?.snapshotRefs ?? ['--']).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>

      </div>
    </section>
  )
}
