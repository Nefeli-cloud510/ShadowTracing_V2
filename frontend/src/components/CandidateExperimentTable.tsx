import { useMemo } from 'react'
import { experimentTypeLabel } from './ExperimentMetrics'

type NumericValue = number | string | { value?: number | string } | undefined

type CandidateExperimentTableProps = {
  candidates: any[]
  treeNodes?: any[]
  uncertaintyRecords?: any[]
  displayText?: (text?: string) => string
  highlightedIndex?: number
}

const EFFECT_LABELS: Record<string, string> = {
  positive: '预期正增益',
  negative: '预期负增益',
  near_zero: '预期近零',
  zero: '预期零增益',
}

type HypothesisRow = {
  id: string
  label: string
  statement: string
  prediction: string
}

function toNumber(value: NumericValue): number {
  if (typeof value === 'number') {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : 0
  }
  if (value && typeof value === 'object') {
    return toNumber(value.value)
  }
  return 0
}

function toDisplayList(values: unknown, displayText: (text?: string) => string): string {
  const list = Array.isArray(values) ? (values as string[]) : []
  if (list.length === 0) {
    return '未配置'
  }
  return list.map((item) => displayText(item) || item).join('、')
}

function getPredictionText(prediction?: any): string {
  if (!prediction) {
    return '实验后按实际 Δ 结果判定'
  }
  const effect = EFFECT_LABELS[String(prediction.expected_effect ?? '')] ?? String(prediction.expected_effect ?? '')
  const range = Array.isArray(prediction.expected_range) && prediction.expected_range.length >= 2
    ? ` [${Number(prediction.expected_range[0]).toFixed(3)}, ${Number(prediction.expected_range[1]).toFixed(3)}]`
    : ''
  return `${effect}${range}`
}

function getFixedSettings(design?: any): string {
  if (!design) {
    return '除所列自变量外保持对照组设置一致'
  }
  const parts: string[] = []
  if (design.forecast_horizon_days) parts.push(`预测超前 ${design.forecast_horizon_days} 天`)
  if (design.past_lag_days) parts.push(`历史滞后 ${design.past_lag_days} 天`)
  if (design.window_size) parts.push(`时间窗口 ${design.window_size} 天`)
  if (design.control_lag_days && design.control_lag_days !== design.treatment_lag_days) {
    parts.push(`对照组滞后 ${design.control_lag_days} 天`)
  }
  if (design.treatment_lag_days && design.treatment_lag_days !== design.control_lag_days) {
    parts.push(`实验组滞后 ${design.treatment_lag_days} 天`)
  }
  if (design.lags && typeof design.lags === 'object') {
    for (const [feature, days] of Object.entries(design.lags as Record<string, unknown>)) {
      if (Array.isArray(days) && days.length > 0) {
        parts.push(`${feature} 滞后 ${days.join('/')} 天`)
      }
    }
  }
  return parts.length > 0 ? parts.join('；') : '除所列自变量外保持对照组设置一致'
}

function getProbeAxis(candidate?: any): string {
  const design = candidate?.design ?? {}
  const notes = Array.isArray(design.notes) ? (design.notes as string[]) : []
  const probeRaw = String(design.probe_axis ?? '').trim()
  const focusRaw = String(design.display_design_focus ?? design.design_focus ?? '').trim()
  const axisLabels: Record<string, string> = {
    incremental_gain: '增量增益轴',
    physical_path: '物理路径轴',
    lead_time: '超前窗口轴',
    robustness: '稳健性轴',
  }

  if (axisLabels[probeRaw]) {
    return axisLabels[probeRaw]
  }

  const probeNote = notes.find((note: string) => /^probe_[^:]*:/.test(note))
  if (probeNote) {
    const axisName = /^probe_([^:]*):/.exec(probeNote)?.[1] ?? ''
    if (axisLabels[axisName]) {
      return axisLabels[axisName]
    }
    if (axisName.includes('conditioned_contribution')) {
      return focusRaw ? `条件增量贡献探测（${focusRaw}）` : '条件增量贡献探测'
    }
  }

  return focusRaw ? focusRaw : '区分性对照'
}

function formatEvaluationMetrics(design?: any): string {
  const raw = Array.isArray(design?.evaluation_metrics)
    ? (design.evaluation_metrics as string[]).filter(
        (item) => !/^\s*MAE\s*$/i.test(String(item).trim()) && !/真实值-预测值/.test(String(item))
      )
    : []
  const metrics = raw.length > 0 ? raw : ['Pearson_r', 'RMSE']
  return `${metrics.join('、')}、真实值-预测值对比图、散点图`
}

export function CandidateExperimentTable({
  candidates,
  treeNodes = [],
  uncertaintyRecords = [],
  displayText = (text?: string) => text ?? '',
  highlightedIndex = 0,
}: CandidateExperimentTableProps) {
  const nodeMap = useMemo(() => {
    const map = new Map<string, any>()
    for (const node of treeNodes) {
      if (node?.hypothesis_id || node?.id) {
        map.set(node.hypothesis_id ?? node.id, node)
      }
    }
    return map
  }, [treeNodes])

  const uncertaintyMap = useMemo(() => {
    const map = new Map<string, any>()
    for (const record of uncertaintyRecords) {
      if (record?.uncertainty_id) {
        map.set(record.uncertainty_id, record)
      }
    }
    return map
  }, [uncertaintyRecords])

  if (candidates.length === 0) {
    return <p>暂无可展示的候选实验数据。</p>
  }

  return (
    <div className="candidate-table">
      {candidates.map((candidate, index) => {
        const design = candidate?.design ?? {}
        const experimentType = candidate?.experiment_type ?? candidate?.type
        const typeLabel = experimentTypeLabel(experimentType)
        const target = displayText(design.display_target ?? design.target) || design.display_target || '待配置'
        const control = design.display_control?.length
          ? toDisplayList(design.display_control, displayText)
          : toDisplayList(design.control, displayText)
        const treatment = design.display_treatment?.length
          ? toDisplayList(design.display_treatment, displayText)
          : toDisplayList(design.treatment, displayText)
        const modelFamilies = Array.isArray(design.model_families) ? design.model_families : []
        const modelName = modelFamilies.length > 0
          ? modelFamilies.map((name: string) => String(name).replace(/^./, (char) => char.toUpperCase())).join(' / ')
          : (() => {
              const modelSource = design.model
              if (typeof modelSource === 'string') {
                return modelSource.replace(/（执行协议固化）.*$/, '').trim()
              }
              if (typeof modelSource === 'object' && modelSource?.name) {
                return String(modelSource.name).replace(/（执行协议固化）.*$/, '').trim()
              }
              return 'ElasticNet'
            })()
        const evaluationMetrics = formatEvaluationMetrics(design)
        const testedHypotheses = Array.isArray(candidate?.tested_hypotheses) ? candidate.tested_hypotheses : []
        const predictions = candidate?.hypothesis_predictions ?? {}
        const discriminationRows: HypothesisRow[] = testedHypotheses.map((id: string, rowIndex: number) => {
          const node = nodeMap.get(id) ?? {}
          const rawLabel = node.display_hypothesis_id || (node.level ? `H${node.level}` : '')
          const label = rawLabel || `假设 ${rowIndex + 1}`
          return {
            id,
            label,
            statement: displayText(node.statement) || node.statement || `未检索到假设陈述（${id}）`,
            prediction: getPredictionText(predictions[id]),
          }
        })
        const relatedUncertainties = Array.isArray(candidate?.related_uncertainties)
          ? candidate.related_uncertainties
          : []
        const purpose = displayText(candidate?.purpose) || candidate?.purpose || displayText(candidate?.scientific_question) || candidate?.scientific_question || '当前候选实验旨在推进核心科学问题。'
        const distinguishingInsight = displayText(candidate?.distinguishing_insight) || candidate?.distinguishing_insight || ''
        const valueAnalysis = displayText(candidate?.value_analysis) || candidate?.value_analysis || ''
        const valueAnalysisBlocks = valueAnalysis.split(/\n+/).map((block: string) => block.trim()).filter(Boolean)
        const priorFinding = candidate?.responded_prior_finding ?? design.responded_prior_finding
        const uncertaintyFocus = candidate?.target_uncertainty_focus ?? design.target_uncertainty_focus
        const changedFactors = candidate?.changed_factors ?? design.changed_factors
        const heldFixedFactors = candidate?.held_fixed_factors ?? design.held_fixed_factors
        const branchEntries = typeof (candidate?.expected_outcome_branches ?? design.expected_outcome_branches) === 'object'
          ? Object.entries(candidate?.expected_outcome_branches ?? design.expected_outcome_branches ?? {})
          : []
        const exploratory = candidate?.exploratory ?? design.exploratory

        return (
          <article
            key={candidate.experiment_id ?? `candidate-${index}`}
            className={`candidate-table__experiment ${index === highlightedIndex ? 'candidate-table__experiment--recommended' : ''}`}
          >
            <header className="candidate-table__header">
              <div>
                <span className="candidate-table__eyebrow">{typeLabel}</span>
                <h3 className="candidate-table__title">{candidate.experiment_id ?? `候选实验 ${index + 1}`}</h3>
              </div>
              <div className="candidate-table__metrics">
                <span className="candidate-table__metric candidate-table__metric--utility">U(E) {toNumber(candidate.utility_score).toFixed(3)}</span>
                <span className="candidate-table__metric">预期区分度 {toNumber(candidate.estimated_information_gain).toFixed(3)}</span>
                <span className="candidate-table__metric">PG {toNumber(candidate.estimated_performance_gain).toFixed(3)}</span>
                <span className="candidate-table__metric">Risk {toNumber(candidate.estimated_risk).toFixed(3)}</span>
                <span className="candidate-table__metric">Cost {toNumber(candidate.estimated_cost).toFixed(3)}</span>
              </div>
            </header>

            <section className="candidate-table__section">
              <span className="candidate-table__section-label">实验目的</span>
              <p>{purpose}</p>
            </section>

            <section className="candidate-table__section">
              <span className="candidate-table__section-label">对应不确定性</span>
              {relatedUncertainties.length > 0 ? (
                <ul className="candidate-table__list candidate-table__list--uncertainty">
                  {relatedUncertainties.map((id: string) => {
                    const record = uncertaintyMap.get(id) ?? {}
                    const question = displayText(record.question ?? record.description) || record.question || record.description || '未命名不确定性'
                    return (
                      <li key={id}>
                        <span className="candidate-table__id-chip">{id}</span>
                        <span>{question}</span>
                      </li>
                    )
                  })}
                </ul>
              ) : (
                <p>{experimentType === 'baseline_benchmark' ? '当前为基准对照候选实验，不直接承接对应不确定性。' : '当前候选未直接承接对应不确定性。'}</p>
              )}
            </section>

            <section className="candidate-table__section candidate-table__section--trace">
              <span className="candidate-table__section-label">实验溯源与预登记</span>
              <div className="candidate-table__grid">
                <div className="candidate-table__row">
                  <strong>所回应的上一轮发现</strong>
                  <p>{displayText(priorFinding) || priorFinding || '本轮为首轮实验：以当前科学不确定性为基线。'}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>要解决的不确定性</strong>
                  <p>{displayText(uncertaintyFocus) || uncertaintyFocus || '见上方对应不确定性条目。'}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>相对上一轮改变的因素</strong>
                  <p>{toDisplayList(changedFactors, displayText)}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>保持固定的条件</strong>
                  <p>{toDisplayList(heldFixedFactors, displayText)}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>预登记结果分支与下一步行动</strong>
                  {branchEntries.length > 0 ? (
                    <ul className="candidate-table__list candidate-table__list--branches">
                      {branchEntries.map(([key, action]) => (
                        <li key={key}>
                          <span className="candidate-table__branch-key">{key}</span>
                          <span>{displayText(String(action)) || String(action)}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>实验完成后按实际 CI 结果更新下一步行动。</p>
                  )}
                </div>
                <div className="candidate-table__row">
                  <strong>实验性质</strong>
                  <p>
                    {exploratory
                      ? <span className="candidate-table__badge candidate-table__badge--exploratory">探索性实验（多因素同时变化，不作单因素归因）</span>
                      : <span className="candidate-table__badge candidate-table__badge--focused">单因素复核 / 验证实验</span>}
                  </p>
                </div>
              </div>
            </section>

            <section className="candidate-table__section">
              <span className="candidate-table__section-label">可验证假设</span>
              <div className="candidate-table__rows">
                {discriminationRows.map((row) => (
                  <div key={row.id} className="candidate-table__row candidate-table__row--hypothesis">
                    <div className="candidate-table__row-label">
                      <strong>{row.label}</strong>
                    </div>
                    <div className="candidate-table__hypothesis-body">
                      <p>
                        <span className="candidate-table__sub-label">假设陈述</span>
                        <br />
                        {row.statement}
                      </p>
                      <p>
                        <span className="candidate-table__sub-label">实验预期</span>
                        <br />
                        {row.prediction}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="candidate-table__section">
              <span className="candidate-table__section-label">区分性思路</span>
              <p>{distinguishingInsight || '尚未生成，正在等待候选实验 LLM 撰稿。'}</p>
            </section>

            <section className="candidate-table__section">
              <span className="candidate-table__section-label">实验设计</span>
              <div className="candidate-table__grid">
                <div className="candidate-table__row">
                  <strong>因变量</strong>
                  <p>{target}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>自变量·对照组</strong>
                  <p>{control}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>自变量·实验组</strong>
                  <p>{treatment}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>控制变量</strong>
                  <p>{getFixedSettings(design)}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>区分焦点</strong>
                  <p>{getProbeAxis(candidate)}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>模型</strong>
                  <p>{modelName}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>评价指标</strong>
                  <p>{evaluationMetrics}</p>
                </div>
                <div className="candidate-table__row">
                  <strong>区分阈值</strong>
                  {discriminationRows.length > 0 ? (
                    <ul className="candidate-table__list candidate-table__list--condensed">
                      {discriminationRows.map((row) => (
                        <li key={row.id}>
                          <strong>{row.label}</strong>
                          <span>{row.prediction}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>实验后按实际 Δ 结果判定</p>
                  )}
                </div>
              </div>
            </section>

            <section className="candidate-table__section">
              <span className="candidate-table__section-label">实验综合价值数据分析</span>
              {valueAnalysisBlocks.length > 0 ? (
                <div className="candidate-table__analysis">
                  {valueAnalysisBlocks.map((block: string, blockIndex: number) => (
                    <p key={blockIndex}>{block}</p>
                  ))}
                </div>
              ) : (
                <p>尚未生成，重新生成候选实验后由 LLM 输出实验综合价值数据分析。</p>
              )}
            </section>
          </article>
        )
      })}
    </div>
  )
}
