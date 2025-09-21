#!/bin/bash
# Playwright MCP Helper Functions

# Test if Playwright MCP server is running
pw_mcp_test() {
    echo "🎭 Testing Playwright MCP Server connection..."
    local response=$(echo '{"jsonrpc":"2.0","id":"test","method":"playwright_status","params":{}}' | nc -w 2 localhost 3003 2>/dev/null)
    if [ -n "$response" ]; then
        echo "✅ Playwright MCP Server is running"
        echo "$response" | jq . 2>/dev/null || echo "$response"
    else
        echo "❌ Playwright MCP Server is not responding"
        echo "💡 Start it with: supervisorctl start playwright-mcp-server"
    fi
}

# Launch headed browser for debugging
pw_debug() {
    echo "🎭 Launching Playwright in headed mode for debugging..."
    PWDEBUG=1 npx playwright test "$@"
}

# Record a new test
pw_record() {
    local url="${1:-https://playwright.dev}"
    local output="${2:-recorded-test.spec.js}"
    echo "🎭 Recording browser actions..."
    echo "📝 Output file: $output"
    echo "🌐 Starting URL: $url"
    npx playwright codegen "$url" -o "$output"
}

# Run specific browser test
pw_run_browser() {
    local browser="${1:-chromium}"
    shift
    echo "🎭 Running tests in $browser..."
    npx playwright test --project="$browser" "$@"
}

# Take screenshot of URL
pw_screenshot() {
    local url="${1:-https://playwright.dev}"
    local output="${2:-screenshot.png}"
    echo "📸 Taking screenshot of $url..."
    npx playwright screenshot "$url" "$output"
    echo "✅ Screenshot saved to: $output"
}

# Check browser installations
pw_browsers() {
    echo "🎭 Installed Playwright browsers:"
    npx playwright install --list
    echo ""
    echo "📁 Browser binaries location:"
    echo "$(npx playwright install --print-browser-dir)"
}

# Generate PDF from webpage
pw_pdf() {
    local url="${1:-https://playwright.dev}"
    local output="${2:-page.pdf}"
    echo "📄 Generating PDF from $url..."
    npx playwright pdf "$url" "$output"
    echo "✅ PDF saved to: $output"
}

# Accessibility audit
pw_accessibility() {
    local url="${1:-https://playwright.dev}"
    cat > /tmp/pw-accessibility-test.js << EOF
const { chromium } = require('playwright');
const { injectAxe, checkA11y } = require('axe-playwright');

(async () => {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    await page.goto('$url');
    await injectAxe(page);
    const results = await checkA11y(page);
    console.log('Accessibility violations:', results);
    await browser.close();
})();
EOF
    node /tmp/pw-accessibility-test.js
    rm /tmp/pw-accessibility-test.js
}

# Show all Playwright MCP commands
pw_mcp_help() {
    echo "🎭 Playwright MCP Server Commands"
    echo "================================="
    echo ""
    echo "Server Management:"
    echo "  playwright-server    - Start MCP server"
    echo "  pw_mcp_test         - Test MCP server connection"
    echo ""
    echo "Testing & Recording:"
    echo "  playwright-test     - Run all tests"
    echo "  pw_debug           - Run tests in debug mode"
    echo "  pw_record [url]    - Record browser actions"
    echo "  pw_run_browser     - Run tests in specific browser"
    echo ""
    echo "Utilities:"
    echo "  pw_screenshot [url] - Take webpage screenshot"
    echo "  pw_pdf [url]       - Generate PDF from webpage"
    echo "  pw_browsers        - List installed browsers"
    echo "  pw_accessibility   - Run accessibility audit"
    echo ""
    echo "MCP Tools Available:"
    echo "  • playwright_navigate"
    echo "  • playwright_screenshot"
    echo "  • playwright_click"
    echo "  • playwright_fill"
    echo "  • playwright_evaluate"
    echo "  • playwright_wait_for_selector"
    echo "  • playwright_get_content"
    echo ""
}

# Auto-load message
echo "🎭 Playwright MCP helpers loaded. Type 'pw_mcp_help' for available commands."