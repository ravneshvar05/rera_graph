import json

html_top = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Graph Plan</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #24292f; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }
        th, td { border: 1px solid #d0d7de; padding: 8px 13px; text-align: left; }
        th { background-color: #f6f8fa; font-weight: 600; }
        tr:nth-child(2n) { background-color: #f6f8fa; }
        pre { background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; font-size: 85%; }
        code { font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace; background-color: rgba(175, 184, 193, 0.2); padding: 0.2em 0.4em; border-radius: 6px; font-size: 85%; }
        pre code { background-color: transparent; padding: 0; }
        blockquote { border-left: 0.25em solid #d0d7de; margin: 0; padding: 0 1em; color: #57606a; }
        h1, h2, h3 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; margin-top: 24px; margin-bottom: 16px; font-weight: 600; }
        .mermaid { text-align: center; margin: 20px 0; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        
        // Define custom renderer
        const renderer = new marked.Renderer();
        const origCode = renderer.code.bind(renderer);
        renderer.code = function({text, lang, escaped}) {
            if (lang === 'mermaid') {
                return '<div class="mermaid">\\n' + text + '\\n</div>';
            }
            return origCode({text, lang, escaped});
        };
        marked.setOptions({ renderer: renderer });
        
        const rawMd = document.getElementById('markdown-content').textContent;
        document.getElementById('content').innerHTML = marked.parse(rawMd);
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
    </script>
</head>
<body>
    <div id="content"></div>
    <script id="markdown-content" type="text/plain">"""

html_bottom = """</script>
</body>
</html>"""

with open('knowledge_graph_plan.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# Make sure we don't accidentally close the script tag
md_content = md_content.replace('</script>', '<\\/script>')

with open('knowledge_graph_plan.html', 'w', encoding='utf-8') as f:
    f.write(html_top)
    f.write(md_content)
    f.write(html_bottom)

print("Generated HTML successfully!")
