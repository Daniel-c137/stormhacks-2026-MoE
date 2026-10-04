"use client";

import { identity } from "@moe/contracts";
import Link from "next/link";
import { useCallback, useRef, useState } from "react";
import { useTeam } from "@/components/AuthProvider";
import { TeamSettingsForm } from "@/components/settings/TeamSettingsForm";
import { Avatar } from "@/components/ui/Avatar";
import { Icon } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { useDismiss } from "@/hooks/useDismiss";
import { signOut } from "@/lib/auth";

/** Brand (identity.product_name), account menu and settings. */
export function AppHeader() {
  const { me, email } = useTeam();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const account = useRef<HTMLDivElement>(null);
  useDismiss(
    account,
    menuOpen,
    useCallback(() => setMenuOpen(false), []),
  );

  return (
    <>
      <header className="topbar">
        <Link className="brand" href="/" aria-label={`${identity.product_name}, meetings home`}>
          <span className="mark-slot">
            <Mark size={40} />
          </span>
          <span className="wordmark">{identity.product_name}</span>
        </Link>
        <div className="topbar-right">
          <div className="account" ref={account}>
            <button
              type="button"
              className="account-btn"
              onClick={() => setMenuOpen((open) => !open)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              aria-label={`Account: ${me.name}`}
              title={me.name}
            >
              <Avatar person={me} me />
            </button>
            {menuOpen && (
              <div className="menu" role="menu" aria-label="Account">
                <div className="account-who">
                  <b>{me.name}</b>
                  {email && <span>{email}</span>}
                </div>
                <Link className="menu-item" role="menuitem" href="/tasks" onClick={() => setMenuOpen(false)}>
                  <Icon name="list-checks" />
                  Tasks
                </Link>
                <button type="button" className="menu-item" role="menuitem" onClick={signOut}>
                  <Icon name="log-out" />
                  Sign out
                </button>
              </div>
            )}
          </div>
          <button type="button" className="icon-btn" onClick={() => setSettingsOpen(true)} aria-label="Settings" title="Settings">
            <Icon name="settings" />
          </button>
        </div>
      </header>
      {/* Outside the header: its backdrop filter would trap a fixed overlay inside it. */}
      {settingsOpen && <TeamSettingsForm onClose={() => setSettingsOpen(false)} />}
    </>
  );
}
