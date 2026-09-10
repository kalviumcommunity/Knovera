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
const REGISTERED_USERS_KEY = 'knovera_registered_users';

export interface RegisteredAccount {
  name: string;
  email: string;
  password: string;
  organization: string;
}

export const DEMO_CREDENTIALS = {
  user: {
    email: 'user@knovera.ai',
    password: 'user123',
    name: 'Alex Johnson',
    organization: 'Enterprise Risk & Operations',
    avatarInitials: 'AJ',
  },
  admin: {
    email: 'admin@knovera.ai',
    password: 'admin123',
    name: 'K Jayanth (Admin)',
    organization: 'Knovera Security Operations',
    avatarInitials: 'KJ',
  },
};

export const getStoredAuth = (): AuthUser | null => {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
};

export const setStoredAuth = (user: AuthUser | null): void => {
  if (typeof window === 'undefined') return;
  try {
    if (user) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch (e) {
    console.error('Failed to update knovera_auth_session', e);
  }
};

export const getRegisteredUsers = (): RegisteredAccount[] => {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(REGISTERED_USERS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

export const registerUser = async (
  name: string,
  email: string,
  password: string,
  organization?: string
): Promise<AuthUser> => {
  await new Promise((resolve) => setTimeout(resolve, 500));

  const trimmedName = name.trim();
  const trimmedEmail = email.trim().toLowerCase();

  if (!trimmedName || trimmedName.length < 2) {
    throw new Error('Please enter your full name (at least 2 characters).');
  }

  if (!trimmedEmail || !trimmedEmail.includes('@') || !trimmedEmail.includes('.')) {
    throw new Error('Please enter a valid email address.');
  }

  if (!password || password.length < 4) {
    throw new Error('Password must be at least 4 characters long.');
  }

  const existing = getRegisteredUsers();
  if (existing.some((u) => u.email === trimmedEmail) || trimmedEmail === DEMO_CREDENTIALS.user.email) {
    throw new Error('An account with this email already exists. Please sign in instead.');
  }

  const org = organization?.trim() || 'Knovera Workspace';
  const newAccount: RegisteredAccount = {
    name: trimmedName,
    email: trimmedEmail,
    password,
    organization: org,
  };

  if (typeof window !== 'undefined') {
    localStorage.setItem(REGISTERED_USERS_KEY, JSON.stringify([...existing, newAccount]));
  }

  const initials = trimmedName
    .split(' ')
    .filter(Boolean)
    .map((n) => n[0])
    .join('')
    .substring(0, 2)
    .toUpperCase() || 'U';

  const user: AuthUser = {
    id: `usr_${Date.now()}`,
    name: trimmedName,
    email: trimmedEmail,
    role: 'user',
    organization: org,
    avatarInitials: initials,
  };

  setStoredAuth(user);
  return user;
};

export const loginAsUser = async (email: string, password: string): Promise<AuthUser> => {
  await new Promise((resolve) => setTimeout(resolve, 400));

  const cleanEmail = email.trim().toLowerCase();

  // Check demo user
  if (cleanEmail === DEMO_CREDENTIALS.user.email && password === DEMO_CREDENTIALS.user.password) {
    const user: AuthUser = {
      id: `usr_demo`,
      name: DEMO_CREDENTIALS.user.name,
      email: cleanEmail,
      role: 'user',
      organization: DEMO_CREDENTIALS.user.organization,
      avatarInitials: DEMO_CREDENTIALS.user.avatarInitials,
    };
    setStoredAuth(user);
    return user;
  }

  // Check custom registered users
  const registered = getRegisteredUsers();
  const matched = registered.find((u) => u.email === cleanEmail && u.password === password);
  if (matched) {
    const initials = matched.name
      .split(' ')
      .filter(Boolean)
      .map((n) => n[0])
      .join('')
      .substring(0, 2)
      .toUpperCase() || 'U';

    const user: AuthUser = {
      id: `usr_${Date.now()}`,
      name: matched.name,
      email: matched.email,
      role: 'user',
      organization: matched.organization,
      avatarInitials: initials,
    };
    setStoredAuth(user);
    return user;
  }

  // Accept general enterprise email format if valid
  if (cleanEmail.includes('@') && password.length >= 4) {
    const user: AuthUser = {
      id: `usr_${Date.now()}`,
      name: cleanEmail.split('@')[0].replace('.', ' ').replace(/^./, (str) => str.toUpperCase()),
      email: cleanEmail,
      role: 'user',
      organization: 'Knovera Enterprise',
      avatarInitials: cleanEmail.substring(0, 2).toUpperCase(),
    };
    setStoredAuth(user);
    return user;
  }

  throw new Error('Invalid email or password. You can sign up for a new account or use demo: user@knovera.ai / user123');
};

export const loginAsAdmin = async (
  email: string,
  password: string,
  mfaCode?: string
): Promise<AuthUser> => {
  await new Promise((resolve) => setTimeout(resolve, 500));

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
