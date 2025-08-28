import { Button } from "@/components/ui/button";
import { javascript } from "@codemirror/lang-javascript";
import { json } from "@codemirror/lang-json";
import { python } from "@codemirror/lang-python";
import { StreamLanguage } from "@codemirror/language";
import { shell } from "@codemirror/legacy-modes/mode/shell";
import { vscodeDark } from "@uiw/codemirror-theme-vscode";
import CodeMirror from "@uiw/react-codemirror";
import { Check, Copy } from "lucide-react";
import { useState } from "react";

interface CodeSnippetProps {
  code: string;
  language: string;
  title?: string;
  showLineNumbers?: boolean;
}

// Map language strings to CodeMirror language extensions
const languageExtensions = {
  javascript: javascript({ jsx: true, typescript: true }),
  typescript: javascript({ jsx: true, typescript: true }),
  jsx: javascript({ jsx: true }),
  tsx: javascript({ jsx: true, typescript: true }),
  json: json(),
  python: python(),
  shell: StreamLanguage.define(shell),
  sh: StreamLanguage.define(shell),
  bash: StreamLanguage.define(shell),
};

export function CodeSnippet({
  code,
  language,
  title,
  showLineNumbers = true,
}: CodeSnippetProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Get the correct language extension, or an empty array for plain text
  const extensions = [
    languageExtensions[
      language.toLowerCase() as keyof typeof languageExtensions
    ] || [],
  ];

  return (
    <div className="relative rounded-lg overflow-hidden border border-border">
      {/* The header is identical to your original component */}
      <div className="flex items-center justify-between px-4 py-2 bg-muted">
        <span className="text-sm text-muted-foreground">
          {title || language}
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          className="h-8 text-muted-foreground hover:text-foreground"
        >
          {copied ? (
            <>
              <Check className="h-4 w-4 mr-1" />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy className="h-4 w-4 mr-1" />
              <span>Copy</span>
            </>
          )}
        </Button>
      </div>

      {/* The SyntaxHighlighter is replaced with CodeMirror */}
      <div className="bg-card text-card-foreground p-4 overflow-x-auto">
        <CodeMirror
          value={code}
          extensions={extensions}
          theme={vscodeDark}
          editable={false}
          basicSetup={{
            lineNumbers: showLineNumbers,
            foldGutter: false, // Hides the code folding gutter
            drawSelection: false, // Hides the cursor/selection
            highlightActiveLine: false,
            highlightActiveLineGutter: false,
          }}
        />
      </div>
    </div>
  );
}
