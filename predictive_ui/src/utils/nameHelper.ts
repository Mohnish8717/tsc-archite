export function cleanPersonaName(name: string): string {
  if (!name) return '';
  // 1. Strip trailing numeric suffix like _0, _1, _17
  let clean = name.replace(/_\d+$/, '');
  // 2. Strip corporate prefixes from the beginning
  clean = clean.replace(/^(Slack|Oasis|Company|Alpha)_/i, '');
  return clean;
}
