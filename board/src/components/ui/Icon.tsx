import {
  Archive,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Calendar,
  CalendarCheck,
  CalendarClock,
  Captions,
  Check,
  ChevronDown,
  ChevronUp,
  Circle,
  CircleCheck,
  CircleDashed,
  CircleDot,
  Clock,
  Copy,
  Ellipsis,
  FileCode,
  FilePenLine,
  FileText,
  GitCompare,
  Hand,
  History,
  ImageUp,
  Info,
  Link2,
  ListChecks,
  LoaderCircle,
  Lock,
  LogIn,
  LogOut,
  type LucideIcon,
  MessageCircle,
  MessageSquare,
  MessageSquareText,
  Mic,
  MicOff,
  MonitorUp,
  MonitorX,
  PanelRightClose,
  Pencil,
  PhoneOff,
  Play,
  Plus,
  Radio,
  RotateCcw,
  Save,
  ScrollText,
  Search,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
  SquareCheck,
  Trash2,
  TriangleAlert,
  Undo2,
  Upload,
  UserPlus,
  Users,
  Video,
  VideoOff,
  Volume2,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";
import type { CSSProperties, SVGProps } from "react";

// The code hosts' and Jira's marks are not part of the icon set. GitLab's and Jira's paths are
// Simple Icons 15.22 (simpleicons.org, CC0-1.0).
function Github(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.1c-3.2.7-3.88-1.36-3.88-1.36-.52-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a10.9 10.9 0 0 1 5.78 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.69 5.39-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z" />
    </svg>
  );
}

function Gitlab(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="m23.6004 9.5927-.0337-.0862L20.3.9814a.851.851 0 0 0-.3362-.405.8748.8748 0 0 0-.9997.0539.8748.8748 0 0 0-.29.4399l-2.2055 6.748H7.5375l-2.2057-6.748a.8573.8573 0 0 0-.29-.4412.8748.8748 0 0 0-.9997-.0537.8585.8585 0 0 0-.3362.4049L.4332 9.5015l-.0325.0862a6.0657 6.0657 0 0 0 2.0119 7.0105l.0113.0087.03.0213 4.976 3.7264 2.462 1.8633 1.4995 1.1321a1.0085 1.0085 0 0 0 1.2197 0l1.4995-1.1321 2.4619-1.8633 5.006-3.7489.0125-.01a6.0682 6.0682 0 0 0 2.0094-7.003z" />
    </svg>
  );
}

function Jira(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M11.571 11.513H0a5.218 5.218 0 0 0 5.232 5.215h2.13v2.057A5.215 5.215 0 0 0 12.575 24V12.518a1.005 1.005 0 0 0-1.005-1.005zm5.723-5.756H5.736a5.215 5.215 0 0 0 5.215 5.214h2.129v2.058a5.218 5.218 0 0 0 5.215 5.214V6.758a1.001 1.001 0 0 0-1.001-1.001zM23.013 0H11.455a5.215 5.215 0 0 0 5.215 5.215h2.129v2.057A5.215 5.215 0 0 0 24 12.483V1.005A1.001 1.001 0 0 0 23.013 0Z" />
    </svg>
  );
}

const ICONS = {
  archive: Archive,
  "arrow-down": ArrowDown,
  "arrow-left": ArrowLeft,
  "arrow-right": ArrowRight,
  "arrow-up": ArrowUp,
  calendar: Calendar,
  "calendar-check": CalendarCheck,
  "calendar-clock": CalendarClock,
  captions: Captions,
  check: Check,
  "chevron-down": ChevronDown,
  "chevron-up": ChevronUp,
  circle: Circle,
  "circle-check": CircleCheck,
  "circle-dashed": CircleDashed,
  "circle-dot": CircleDot,
  clock: Clock,
  copy: Copy,
  ellipsis: Ellipsis,
  "file-code": FileCode,
  "file-pen-line": FilePenLine,
  "file-text": FileText,
  "git-compare": GitCompare,
  github: Github,
  gitlab: Gitlab,
  hand: Hand,
  history: History,
  "image-up": ImageUp,
  info: Info,
  jira: Jira,
  link: Link2,
  "list-checks": ListChecks,
  "loader-circle": LoaderCircle,
  lock: Lock,
  "log-in": LogIn,
  "log-out": LogOut,
  "message-circle": MessageCircle,
  "message-square": MessageSquare,
  "message-square-text": MessageSquareText,
  mic: Mic,
  "mic-off": MicOff,
  "monitor-up": MonitorUp,
  "monitor-x": MonitorX,
  "panel-right-close": PanelRightClose,
  pencil: Pencil,
  "phone-off": PhoneOff,
  play: Play,
  plus: Plus,
  radio: Radio,
  "rotate-ccw": RotateCcw,
  save: Save,
  "scroll-text": ScrollText,
  search: Search,
  send: Send,
  settings: Settings,
  "shield-check": ShieldCheck,
  sparkles: Sparkles,
  "square-check": SquareCheck,
  "trash-2": Trash2,
  "triangle-alert": TriangleAlert,
  "undo-2": Undo2,
  upload: Upload,
  "user-plus": UserPlus,
  users: Users,
  video: Video,
  "video-off": VideoOff,
  "volume-2": Volume2,
  "wand-sparkles": WandSparkles,
  x: X,
  zap: Zap,
} satisfies Record<string, LucideIcon | typeof Github>;

export type IconName = keyof typeof ICONS;

/** Sized by the surrounding font-size, like the icon font in the design. */
export function Icon({ name, className, style }: { name: IconName; className?: string; style?: CSSProperties }) {
  const Svg = ICONS[name];
  return (
    <i className={`icon-${name}${className ? ` ${className}` : ""}`} aria-hidden="true" style={style}>
      <Svg />
    </i>
  );
}

export function Spinner({ size }: { size?: number }) {
  return <Icon name="loader-circle" className="spin" style={size ? { fontSize: size } : undefined} />;
}
