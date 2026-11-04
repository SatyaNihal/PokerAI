import {test,expect} from '@playwright/test';
import type {Game} from '../lib/api';

for(const game of ['kuhn','leduc'])for(const algorithm of ['cfr','cfr_plus','deep_cfr'])for(const seat of [0,1]) {
  test(`complete ${game} against ${algorithm} from seat ${seat}`,async({page},testInfo)=>{
    const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto('/');
    await expect(page.getByRole('button',{name:'Deal a hand'})).toBeEnabled();
    await page.getByLabel('Game',{exact:true}).selectOption(game);
    await page.getByLabel('Opponent',{exact:true}).selectOption(`${game}-${algorithm}`);
    await page.getByLabel('Your position').selectOption(String(seat));
    const [created]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/games')&&r.request().method()==='POST'),page.getByRole('button',{name:'Deal a hand'}).click()]);
    let hand:Game=await created.json();
    expect(created.status()).toBe(201);
    if(game==='kuhn')await expect(page.getByTestId('community')).toHaveCount(0);
    if(game==='leduc'&&algorithm==='cfr_plus'&&seat===0)await page.screenshot({path:testInfo.outputPath('play.png'),fullPage:true});
    let moves=0;
    while(hand.status==='playing') {
      expect(hand).not.toHaveProperty('ai_card');expect(hand).not.toHaveProperty('deck');expect(hand).not.toHaveProperty('rng_state');
      expect(moves++).toBeLessThan(12);
      const action=game==='leduc'?1:Number(Boolean(hand.history.length&&hand.history.at(-1)!.action===1));
      const [response]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/actions')&&r.request().method()==='POST'),page.locator(`button[data-action="${action}"]`).click()]);
      expect(response.status()).toBe(200);hand=await response.json();
    }
    await expect(page.getByTestId('hand-result')).toBeVisible();
    await page.getByRole('button',{name:'View AI decisions'}).click();
    await expect(page.getByRole('region',{name:'Post-game analysis'})).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test('explore observable states and compare all saved methods',async({page},testInfo)=>{
  await page.goto('/#explorer');
  await page.getByLabel('Explorer game').selectOption('leduc');
  await expect(page.locator('.policy-card')).toHaveCount(3);
  await page.getByLabel('Private card',{exact:true}).selectOption('2');
  await page.getByLabel('Public card',{exact:true}).selectOption('0');
  await expect(page.locator('.policy-card')).toHaveCount(3);
  await page.getByLabel('Betting history').selectOption({index:1});
  await expect(page.locator('.policy-card')).toHaveCount(3);
  await page.getByText('Inspect the model input',{exact:true}).click();
  await expect(page.locator('.encoder-details code')).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('explorer.png'),fullPage:true});
  await page.getByLabel('Explorer game').selectOption('kuhn');
  await expect(page.locator('.policy-card')).toHaveCount(3);
  await expect(page.getByLabel('Public card',{exact:true})).toHaveCount(0);
});

test('dashboard shows real recorded curves, compute, losses, and tournament intervals',async({page},testInfo)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/#dashboard');
  await expect(page.getByRole('heading',{name:'Exploitability over training'})).toBeVisible();
  await expect(page.getByRole('img',{name:'Exploitability convergence'})).toBeVisible();
  await page.getByLabel('Compute axis',{exact:true}).selectOption('nodes_touched');
  await page.getByLabel('Compute axis',{exact:true}).selectOption('elapsed_seconds');
  expect(await page.locator('.convergence-panel .recharts-yAxis .recharts-cartesian-axis-tick-value').allTextContents()).not.toContain('0');
  await expect(page.getByRole('heading',{name:'Head-to-head results'})).toBeVisible();
  await expect(page.locator('table').first().locator('tbody tr')).toHaveCount(7);
  await page.getByText('Configuration & provenance',{exact:true}).click();
  await expect(page.locator('.provenance-panel').filter({hasText:'Experiment details'}).locator('pre')).toContainText('code_fingerprint');
  await page.screenshot({path:testInfo.outputPath('dashboard.png'),fullPage:true});
  await page.getByLabel('Dashboard game').selectOption('kuhn');
  await expect(page.locator('table').first().locator('tbody tr')).toHaveCount(7);
  expect(errors).toEqual([]);
});

test('mobile layout has no horizontal overflow across screens',async({page},testInfo)=>{
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Deal a hand'})).toBeEnabled();
  for(const label of ['Play','Decision Explorer','Research Dashboard']) {
    await page.getByRole('button',{name:label,exact:true}).click();
    if(label==='Decision Explorer')await expect(page.locator('.policy-card')).toHaveCount(1);
    if(label==='Research Dashboard')await expect(page.getByRole('heading',{name:'Exploitability over training'})).toBeVisible();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  }
  await page.getByRole('button',{name:'Play',exact:true}).click();
  await page.screenshot({path:testInfo.outputPath('mobile.png'),fullPage:true});
});

test('network failure is visible and retry loads real agents',async({page})=>{
  await page.route('**/agents',route=>route.abort());
  await page.goto('/');
  await expect(page.getByRole('alert',{name:'Request error'})).toBeVisible();
  await page.unroute('**/agents');
  await page.getByRole('button',{name:'Try again'}).click();
  await expect(page.getByRole('button',{name:'Deal a hand'})).toBeEnabled();
});
