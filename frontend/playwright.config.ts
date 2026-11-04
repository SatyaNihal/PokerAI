import {defineConfig} from '@playwright/test';
import path from 'node:path';
const externalURL=process.env.POKER_BASE_URL;

export default defineConfig({
  testDir:'./e2e',
  timeout:30000,
  fullyParallel:false,
  workers:1,
  retries:0,
  reporter:[['list'],['json',{outputFile:process.env.POKER_E2E_REPORT||'test-results/report.json'}]],
  use:{baseURL:externalURL||'http://127.0.0.1:8765',viewport:{width:1440,height:1000},trace:'retain-on-failure',screenshot:'only-on-failure'},
  webServer:externalURL?undefined:{command:'.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765',cwd:path.resolve(__dirname,'..'),url:'http://127.0.0.1:8765/health',reuseExistingServer:!process.env.CI,timeout:60000}
});
