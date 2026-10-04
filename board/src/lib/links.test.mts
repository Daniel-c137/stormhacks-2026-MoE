// node --test: `pnpm --filter board test`
import assert from "node:assert/strict";
import { test } from "node:test";
import { codeHost, jiraIssueUrl } from "./links.ts";

test("a Jira link works whether the site was saved with a scheme or not", () => {
  for (const site of ["dropsubs.atlassian.net", "https://dropsubs.atlassian.net", "https://dropsubs.atlassian.net/", " http://dropsubs.atlassian.net "]) {
    assert.equal(jiraIssueUrl(site, "DS-104"), "https://dropsubs.atlassian.net/browse/DS-104");
  }
  assert.equal(jiraIssueUrl("jira.acme.example/jira/", "AC-1"), "https://jira.acme.example/jira/browse/AC-1");
});

test("no site or no key is no link", () => {
  assert.equal(jiraIssueUrl(null, "DS-1"), null);
  assert.equal(jiraIssueUrl("  ", "DS-1"), null);
  assert.equal(jiraIssueUrl("acme.atlassian.net", null), null);
});

test("code links name their host", () => {
  assert.equal(codeHost("https://github.com/dropsubs/web/blob/abc/lib/pricing.ts#L1-L2"), "github");
  assert.equal(codeHost("https://gitlab.com/dropsubs/infra/-/blob/abc/monitoring/status.yml#L1-2"), "gitlab");
  assert.equal(codeHost("https://gitlab.example.com/a/b/c/-/blob/main/x.py#L3-4"), "gitlab");
  assert.equal(codeHost("not a url"), "github");
});
