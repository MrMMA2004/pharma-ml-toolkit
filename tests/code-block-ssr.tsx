/** SSR guard for the explorer's code theme. Does not simulate browser hydration. */
import { strict as assert } from "node:assert";
import React from "react";
import { renderToString } from "react-dom/server";
import { ThemeProvider } from "next-themes";
import { CodeBlock } from "../src/components/explorer/code-block";

function serverPreStyle(theme: "light" | "dark") {
  const html = renderToString(
    <ThemeProvider forcedTheme={theme} defaultTheme={theme} attribute="class">
      <CodeBlock code="print(1)" language="python" />
    </ThemeProvider>,
  );
  const style = /<pre[^>]*style="([^"]+)"/.exec(html)?.[1];
  assert.ok(style, "syntax highlighter did not render its pre style");
  return style;
}

assert.equal(serverPreStyle("light"), serverPreStyle("dark"),
  "the server markup must not depend on the client theme");
console.log("CodeBlock server light/dark markup matches");
