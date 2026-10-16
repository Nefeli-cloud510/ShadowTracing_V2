const fs = require('node:fs')
const path = require('node:path')

const PWROOT = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'
const { chromium } = require(PWROOT)

const SESSION = path.join(
  process.cwd(),
  'runtime/live_session/sessions/session_20261016_235451_805278',
)
const OUT_DIR = path.join(process.cwd(), 'artifacts/hypothesis_tree_snapshots')

const readJson = (relativePath) => {
  const raw = fs.readFileSync(path.join(SESSION, relativePath), 'utf-8')
  return JSON.parse(raw)
}

const currentTree = readJson('state/hypothesis_tree.json')
const preQuestioningTree = readJson('state/hypothesis_tree.pre_questioning.json')
const roundOneSnapshot = readJson('state/round_snapshots/round_01.json')

const roundOneOverrides = {}
for (const [fileName, value] of Object.entries(roundOneSnapshot.artifacts ?? {})) {
  if (typeof fileName === 'string' && fileName.endsWith('.json')) {
    roundOneOverrides[fileName] = value
  } else if (typeof fileName === 'string') {
    roundOneOverrides[`${fileName}.json`] = value
  }
}

const snapshots = [
  {
    id: 'round01_evaluation_writeback',
    overrides: roundOneOverrides,
    nodeIndex: 4,
    label: 'round 1 evaluation writeback',
  },
  {
    id: 'round02_pre_questioning',
    overrides: { 'hypothesis_tree.json': preQuestioningTree },
    nodeIndex: 0,
    label: 'round 2 pre-questioning',
  },
  {
    id: 'round02_evaluation_writeback',
    overrides: { 'hypothesis_tree.json': currentTree },
    nodeIndex: 0,
    label: 'round 2 evaluation writeback',
  },
]

async function capture(snapshot) {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  })
  const context = await browser.newContext({
    viewport: { width: 1680, height: 1050 },
    deviceScaleFactor: 1.5,
  })
  const page = await context.newPage()
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.log(`[page:error] ${msg.text()}`)
  })
  page.on('pageerror', (err) => console.log(`[page:uncaught] ${err.message}`))

  await page.route('http://127.0.0.1:8765/api/state/**', async (route) => {
    const url = new URL(route.request().url())
    const fileName = decodeURIComponent(url.pathname.split('/').pop() ?? '')
    const override = snapshot.overrides[fileName]
    if (override !== undefined) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json; charset=utf-8',
        body: JSON.stringify(override),
      })
      return
    }
    await route.continue()
  })

  await page.goto('http://localhost:5173/hypotheses', {
    waitUntil: 'domcontentloaded',
  })
  await page.locator('.hypothesis-graph__node').first().waitFor({ state: 'visible', timeout: 20000 })
  await page.waitForTimeout(800)

  const nodes = page.locator('.hypothesis-graph__node')
  const targetNode = nodes.nth(snapshot.nodeIndex)
  await targetNode.click({ force: true, position: { x: 10, y: 10 } })
  try {
    await page.locator('.approval-overlay__panel.hypothesis-node-panel').waitFor({
      state: 'visible',
      timeout: 3000,
    })
  } catch {
    console.log(`[retry:keyboard] ${snapshot.label}`)
    await targetNode.focus()
    await page.keyboard.press('Enter')
    await page.locator('.approval-overlay__panel.hypothesis-node-panel').waitFor({
      state: 'visible',
      timeout: 10000,
    })
  }
  await page.waitForTimeout(600)

  const stateInfo = await page.evaluate(() => {
    const nodeEls = Array.from(document.querySelectorAll('.hypothesis-graph__node'))
    const panel = document.querySelector('.approval-overlay__panel.hypothesis-node-panel')
    const root = document.querySelector('.hypothesis-graph__root strong')
    return {
      nodeCount: nodeEls.length,
      nodeTexts: nodeEls.map((el) => (el.textContent ?? '').replace(/\s+/g, ' ').trim()),
      rootText: root?.textContent?.trim() ?? null,
      panelVisible: !!panel,
      panelText: panel?.textContent?.replace(/\s+/g, ' ').trim().slice(0, 300) ?? null,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
      panelBox: panel?.getBoundingClientRect() && {
        top: Math.round(panel.getBoundingClientRect().top),
        bottom: Math.round(panel.getBoundingClientRect().bottom),
        left: Math.round(panel.getBoundingClientRect().left),
        right: Math.round(panel.getBoundingClientRect().right),
      },
    }
  })
  console.log(`[state] ${snapshot.label}`, JSON.stringify(stateInfo))

  const outPath = path.join(OUT_DIR, `${snapshot.id}.png`)
  await page.screenshot({ path: outPath })
  await context.close()
  await browser.close()
  console.log(`captured ${snapshot.label} -> ${outPath}`)
}

;(async () => {
  for (const snapshot of snapshots) {
    await capture(snapshot)
  }
})()
