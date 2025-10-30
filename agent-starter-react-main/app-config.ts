import type { AppConfig } from './lib/types';

export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'Voice Cloning Agent',
  pageTitle: 'Voice Cloning Agent',
  pageDescription: 'Real-time voice cloning with Speechify',

  supportsChatInput: true,
  supportsVideoInput: false,
  supportsScreenShare: false,
  isPreConnectBufferEnabled: true,

  logo: '/lk-logo.svg',
  accent: '#667eea',
  logoDark: '/lk-logo-dark.svg',
  accentDark: '#764ba2',
  startButtonText: 'Start Voice Cloning',

  agentName: undefined,
};
