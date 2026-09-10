'use client';

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: 'user' | 'admin';
  organization: string;
  avatarInitials: string;
}

const STORAGE_KEY = 'knovera_auth_session';

export const DEMO_CREDENTIALS = {
  user: {
    email: 'user@knovera.ai',
    password: 'user123',
    name: 'Sarah Jenkins',
    role: 'user' as const,
    organization: 'Knovera Enterprise',
    avatarInitials: 'SJ',
  },
  admin: {
    email: 'admin@knovera.ai',
    password: 'admin123',
    name: 'K Jayanth (Admin)',
    role: 'admin' as const,
    organization: 'Knovera Global Security',
    avatarInitials: 'KJ',
  },
};

export const getStoredAuth = (): AuthUser | null => {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
};

export const setStoredAuth = (user: AuthUser | null) => {
  if (typeof window === 'undefined') return;
  if (user) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
};

export const loginAsUser = async (email: string, password: string): Promise<AuthUser> => {
  // Simulate network latency
  await new Promise((resolve) => setTimeout(resolve, 500));

  // Accept demo user or any valid email format with matching password or demo password
  if (
    (email.toLowerCase() === DEMO_CREDENTIALS.user.email && password === DEMO_CREDENTIALS.user.password) ||
    (email.includes('@') && password.length >= 4)
  ) {
    const user: AuthUser = {
      id: `usr_${Date.now()}`,
      name: email.split('@')[0].replace('.', ' ').replace(/^./, (str) => str.toUpperCase()),
      email: email.toLowerCase(),
      role: 'user',
      organization: 'Knovera Enterprise',
      avatarInitials: email.substring(0, 2).toUpperCase(),
    };
    setStoredAuth(user);
    return user;
  }

  throw new Error('Invalid email or password. You can use demo: user@knovera.ai / user123');
};

export const loginAsAdmin = async (
  email: string,
  password: string,
  mfaCode?: string
): Promise<AuthUser> => {
  // Simulate network latency
  await new Promise((resolve) => setTimeout(resolve, 600));

  // Accept demo admin or admin email
  if (
    (email.toLowerCase() === DEMO_CREDENTIALS.admin.email && password === DEMO_CREDENTIALS.admin.password) ||
    (email.toLowerCase().includes('admin') && password.length >= 4)
  ) {
    const admin: AuthUser = {
      id: `adm_${Date.now()}`,
      name: 'K Jayanth (Security Admin)',
      email: email.toLowerCase(),
      role: 'admin',
      organization: 'Knovera Global Security',
      avatarInitials: 'KJ',
    };
    setStoredAuth(admin);
    return admin;
  }

  throw new Error('Invalid administrator credentials. You can use demo: admin@knovera.ai / admin123');
};

export const logout = () => {
  setStoredAuth(null);
};
