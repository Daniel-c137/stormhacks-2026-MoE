import type { CodeSnippet } from "@moe/contracts";
import { Icon } from "@/components/ui/Icon";
import { CODE_HOST_NAME, codeHost } from "@/lib/links";

export interface CodeStageProps {
  snippet: CodeSnippet;
  onClose: () => void;
}

/** A snippet shown to everyone, with its GitHub link and highlighted lines. */
export function CodeStage({ snippet, onClose }: CodeStageProps) {
  const [from, to] = snippet.highlight ?? [0, -1];
  return (
    <div className="code code-stage" role="region" aria-label={`Code: ${snippet.path}`}>
      <div className="code-head">
        <Icon name="file-code" />
        <span className="path">{snippet.repo ? `${snippet.repo}: ${snippet.path}` : snippet.path}</span>
        <span className="rng">
          Lines {snippet.start_line}–{snippet.end_line}
        </span>
        <a className="rng" href={snippet.github_url} target="_blank" rel="noopener noreferrer">
          {CODE_HOST_NAME[codeHost(snippet.github_url)]}
        </a>
        <button type="button" className="icon-btn sm" onClick={onClose} aria-label="Close the code for everyone">
          <Icon name="x" />
        </button>
      </div>
      <pre>
        {snippet.code.split("\n").map((text, i) => {
          const n = snippet.start_line + i;
          return (
            <div key={n} className="code-line" data-hl={n >= from && n <= to}>
              <span className="n">{n}</span>
              <span>{text || " "}</span>
            </div>
          );
        })}
      </pre>
      {snippet.caption && <p className="cap">{snippet.caption}</p>}
    </div>
  );
}
