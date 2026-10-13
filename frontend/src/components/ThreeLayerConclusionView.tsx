import type { ThreeLayerConclusionViewModel } from '../types/timeline'
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
