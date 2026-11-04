import {test,expect} from '@playwright/test';
import type {Game} from '../lib/api';

for(const seat of [0,1])test(`complete limit holdem from seat ${seat} with saved neural policy`,async({page},testInfo)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Deal a hand'})).toBeEnabled();
  await expect(page.getByLabel('Game',{exact:true})).toHaveValue('holdem');
  await expect(page.getByLabel('Opponent',{exact:true})).toHaveValue('holdem-deep_cfr');
  await page.getByLabel('Your position').selectOption(String(seat));
  const [created]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/games')&&r.request().method()==='POST'),page.getByRole('button',{name:'Deal a hand'}).click()]);
  let hand:Game=await created.json();expect(created.status()).toBe(201);
  expect(hand.private_cards).toHaveLength(2);
  await expect(page.locator('.human-seat .playing-card')).toHaveCount(2);
  await expect(page.getByTestId('community').locator('.playing-card')).toHaveCount(5);
  let steps=0,shot=false;
  while(hand.status==='playing'){
    expect(steps++).toBeLessThan(30);
    expect(hand).not.toHaveProperty('ai_cards');expect(hand).not.toHaveProperty('deck');
    if(hand.round===2&&!shot){await page.screenshot({path:testInfo.outputPath(`holdem-seat-${seat}.png`),fullPage:true});shot=true;}
    const [response]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/actions')&&r.request().method()==='POST'),page.locator('button[data-action="1"]').click()]);
    expect(response.status()).toBe(200);hand=await response.json();
  }
  if(seat===0)expect(hand.public_cards).toHaveLength(5);
  await expect(page.getByTestId('hand-result')).toBeVisible();
  await page.getByRole('button',{name:'View AI decisions'}).click();
  await expect(page.getByRole('region',{name:'Post-game analysis'})).toBeVisible();
  await expect(page.locator('.opponent-seat .card-back')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('holdem explorer evaluates edited observable cards',async({page},testInfo)=>{
  await page.goto('/#explorer');
  await expect(page.getByLabel('Explorer game')).toHaveValue('holdem');
  await expect(page.locator('.policy-card')).toHaveCount(1);
  const field=page.getByLabel('Hole card 1',{exact:true});
  const before=await field.inputValue();
  const values=await field.locator('option').evaluateAll(options=>options.map(o=>(o as HTMLOptionElement).value));
  const [response]=await Promise.all([page.waitForResponse(r=>r.url().includes('/agents/holdem-deep_cfr/policy')),field.selectOption(values.find(v=>v!==before&&Number(v)>=32&&Number(v)<=35)!)]);
  expect(response.status()).toBe(200);
  await expect(page.locator('.policy-card')).toHaveCount(1);
  await expect(page.locator('.observation-cards .playing-card[aria-label^="10"]').first()).toBeVisible();
  await page.getByText('Inspect the model input',{exact:true}).click();
  await expect(page.locator('.encoder-details code')).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('holdem-explorer.png'),fullPage:true});
});

test('holdem dashboard shows recorded seed results and approximate evaluation',async({page},testInfo)=>{
  await page.goto('/#dashboard');
  await expect(page.getByRole('heading',{name:'Held-out Hold’em evaluation'})).toBeVisible();
  await expect(page.getByRole('table',{name:'Holdem seed results'}).locator('tbody tr')).toHaveCount(5);
  await page.getByLabel('Holdem compute axis').selectOption('nodes_touched');
  await page.getByLabel('Holdem compute axis').selectOption('elapsed_seconds');
  await expect(page.getByRole('img',{name:'Holdem advantage_loss',exact:true})).toBeVisible();
  await expect(page.getByRole('img',{name:'Holdem strategy_loss',exact:true})).toBeVisible();
  await page.locator('#holdem-study').screenshot({path:testInfo.outputPath('holdem-dashboard.png')});
});

test('holdem table and explorer fit mobile width',async({page},testInfo)=>{
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Deal a hand'})).toBeEnabled();
  await page.getByLabel('Game',{exact:true}).selectOption('holdem');
  await expect(page.getByLabel('Opponent',{exact:true})).toHaveValue('holdem-deep_cfr');
  await page.getByLabel('Your position').selectOption('0');
  await page.getByRole('button',{name:'Deal a hand'}).click();
  await expect(page.getByTestId('community')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await page.screenshot({path:testInfo.outputPath('holdem-mobile.png'),fullPage:true});
  await page.getByRole('button',{name:'Decision Explorer',exact:true}).click();
  await page.getByLabel('Explorer game').selectOption('holdem');
  await expect(page.getByLabel('Hole card 1',{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
});

for(const width of [1440,390])test(`reveal and hide an early-fold board at width ${width}`,async({page,request},testInfo)=>{
  await page.setViewportSize({width,height:1000});
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Deal a hand'})).toBeEnabled();
  await page.getByLabel('Game',{exact:true}).selectOption('holdem');
  await page.getByLabel('Your position').selectOption('0');
  const [created]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/games')&&r.request().method()==='POST'),page.getByRole('button',{name:'Deal a hand'}).click()]);
  const hand:Game=await created.json();
  await expect(page.getByRole('button',{name:'Show remaining cards',exact:true})).toHaveCount(0);
  expect((await request.get(`/games/${hand.id}/runout`)).status()).toBe(403);
  const [folded]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/actions')),page.locator('button[data-action="0"]').click()]);
  const result:Game=await folded.json();
  expect(result.status).toBe('complete');
  expect(result.public_cards).toHaveLength(0);
  const board=page.getByTestId('community');
  await expect(board.locator('.card-back')).toHaveCount(5);
  const [revealed]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/runout')),page.getByRole('button',{name:'Show remaining cards',exact:true}).click()]);
  expect(revealed.status()).toBe(200);
  const runout=await revealed.json();
  expect(runout.public_cards).toHaveLength(5);
  await expect(board.locator('.card-back')).toHaveCount(0);
  await expect(page.locator('.opponent-seat .card-back')).toHaveCount(2);
  await expect(page.getByText('Remaining cards revealed after the hand. Result unchanged.',{exact:true})).toBeVisible();
  expect(await (await request.get(`/games/${hand.id}`)).json()).toEqual(result);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await page.screenshot({path:testInfo.outputPath(`runout-${width}.png`),fullPage:true});
  await page.getByRole('button',{name:'Hide remaining cards',exact:true}).click();
  await expect(board.locator('.card-back')).toHaveCount(5);
  await page.getByRole('button',{name:'Show remaining cards',exact:true}).click();
  await expect(board.locator('.card-back')).toHaveCount(0);
  await page.getByRole('button',{name:'View AI decisions',exact:true}).click();
  await expect(page.getByRole('region',{name:'Post-game analysis'})).toBeVisible();
  await page.getByRole('button',{name:'Play again',exact:true}).click();
  await expect(page.locator('button[data-action="0"]')).toBeVisible();
  await expect(board.locator('.card-back')).toHaveCount(5);
  await expect(page.getByRole('button',{name:'Show remaining cards',exact:true})).toHaveCount(0);
});
