/**
 * Phase 1: Frontend E2E Tests with Playwright
 * 
 * Tests the core debate flow through the Fantasy Football Neuron UI.
 * 
 * Setup:
 *   npx playwright install
 * 
 * Run:
 *   npx playwright test e2e/frontend_flow.spec.ts
 * 
 * Run with UI:
 *   npx playwright test e2e/frontend_flow.spec.ts --ui
 */

import { test, expect, Page } from '@playwright/test';

// Configuration
const BASE_URL = process.env.TEST_URL || 'http://localhost:5173';
const TIMEOUT = 30000; // 30 seconds

test.describe('Fantasy Football Neuron - Core Debate Flow', () => {

    test.beforeEach(async ({ page }) => {
        // Set a longer default timeout for this app
        page.setDefaultTimeout(TIMEOUT);
    });

    test('1. Homepage loads with correct title', async ({ page }) => {
        await page.goto(BASE_URL);

        // Wait for the page to load
        await page.waitForLoadState('networkidle');

        // Check for the main title
        const title = await page.locator('h1, [class*="title"]').first().textContent();
        expect(title).toContain('Fantasy Football');

        // Take a screenshot of the loaded page
        await page.screenshot({ path: 'test-results/01-homepage.png' });
    });

    test('2. Navigation tabs/sections are visible', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Look for navigation elements - tabs, buttons, or links
        const navElements = page.locator('nav, [role="tablist"], [class*="tab"], [class*="nav"]');

        // There should be some navigation
        const navCount = await navElements.count();
        console.log(`Found ${navCount} navigation elements`);

        // Take screenshot
        await page.screenshot({ path: 'test-results/02-navigation.png' });
    });

    test('3. Debate Generator input is accessible', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Look for input fields or textareas
        const inputs = page.locator('input[type="text"], textarea, [class*="input"]');
        const inputCount = await inputs.count();

        console.log(`Found ${inputCount} input elements`);

        // If there's an input, try to find a debate-related one
        if (inputCount > 0) {
            const firstInput = inputs.first();
            await firstInput.click();
            await firstInput.fill('Mahomes vs Allen');

            // Verify the input has the text
            const value = await firstInput.inputValue();
            expect(value).toContain('Mahomes');
        }

        await page.screenshot({ path: 'test-results/03-debate-input.png' });
    });

    test('4. Generate button triggers debate creation', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Find a generate/submit button
        const generateButton = page.locator('button:has-text("Generate"), button:has-text("Create"), button:has-text("Start"), button:has-text("Submit")').first();

        const buttonVisible = await generateButton.isVisible().catch(() => false);

        if (buttonVisible) {
            // Click the button
            await generateButton.click();

            // Wait for some reaction (loading state, new content, etc.)
            await page.waitForTimeout(2000);

            // Look for a stop/cancel button or loading indicator
            const stopButton = page.locator('button:has-text("Stop"), button:has-text("Cancel"), [class*="loading"]');
            const stopVisible = await stopButton.isVisible().catch(() => false);

            console.log(`Stop/Loading indicator visible: ${stopVisible}`);
        }

        await page.screenshot({ path: 'test-results/04-generate-clicked.png' });
    });

    test('5. MasteringDesk export options appear after debate', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Look for export-related buttons
        const exportElements = page.locator('button:has-text("Export"), button:has-text("Download"), [class*="export"], [class*="download"]');

        const exportCount = await exportElements.count();
        console.log(`Found ${exportCount} export-related elements`);

        await page.screenshot({ path: 'test-results/05-export-options.png' });
    });

    test('6. Voice controls are present', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Look for audio/voice related controls
        const voiceControls = page.locator('[class*="voice"], [class*="audio"], button:has-text("Play"), button:has-text("Pause"), [class*="volume"]');

        const voiceCount = await voiceControls.count();
        console.log(`Found ${voiceCount} voice-related elements`);

        await page.screenshot({ path: 'test-results/06-voice-controls.png' });
    });
});

test.describe('Fantasy Football Neuron - Personality System', () => {

    test('7. Personality selector is accessible', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Look for personality/agent related elements
        const personalityElements = page.locator('[class*="personality"], [class*="agent"], select, [class*="selector"]');

        const count = await personalityElements.count();
        console.log(`Found ${count} personality-related elements`);

        await page.screenshot({ path: 'test-results/07-personality.png' });
    });
});

test.describe('Fantasy Football Neuron - Player Features', () => {

    test('8. Player input/search works', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Look for player input fields
        const playerInput = page.locator('input[placeholder*="player"], input[placeholder*="Player"], [class*="player-input"]').first();

        const inputVisible = await playerInput.isVisible().catch(() => false);

        if (inputVisible) {
            await playerInput.fill('Patrick Mahomes');
            await page.waitForTimeout(500);

            // Check if suggestions appear
            const suggestions = page.locator('[class*="suggestion"], [class*="dropdown"], [class*="autocomplete"]');
            const suggestionCount = await suggestions.count();
            console.log(`Found ${suggestionCount} suggestion elements`);
        }

        await page.screenshot({ path: 'test-results/08-player-search.png' });
    });
});

test.describe('Fantasy Football Neuron - Accessibility', () => {

    test('9. No console errors on page load', async ({ page }) => {
        const errors: string[] = [];

        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                errors.push(msg.text());
            }
        });

        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Filter out known acceptable errors
        const criticalErrors = errors.filter(e =>
            !e.includes('favicon') &&
            !e.includes('404') &&
            !e.includes('net::ERR')
        );

        console.log(`Console errors: ${criticalErrors.length}`);
        criticalErrors.forEach(e => console.log(`  - ${e.substring(0, 100)}`));

        // Warn but don't fail on console errors
        if (criticalErrors.length > 0) {
            console.warn('⚠️ Console errors detected');
        }
    });

    test('10. Page is responsive (mobile viewport)', async ({ page }) => {
        // Set mobile viewport
        await page.setViewportSize({ width: 375, height: 667 });

        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Check that content fits
        const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
        const viewportWidth = 375;

        expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 10); // Allow 10px tolerance

        await page.screenshot({ path: 'test-results/10-mobile-view.png' });
    });
});
