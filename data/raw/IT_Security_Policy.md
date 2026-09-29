# Sentinel Global — Information Security & Acceptable Use Policy

**Document ID:** POL-SEC-2024-001  
**Effective Date:** February 15, 2024  
**Version:** 4.0  
**Owner:** Chief Information Security Office (CISO)  

---

## 1. Scope & Objective
This policy governs security protocols regarding corporate data, network access, authentication, and endpoint devices assigned to personnel.

---

## 2. Authentication & Credential Standards

### 2.1 Password Requirements
* Minimum length of **12 characters**.
* Must contain uppercase letters, lowercase letters, numbers, and at least one special symbol (!@#$%^&*).
* Mandatory password change interval is **90 days**.
* Passwords cannot match any of the previous 6 historical passwords.

### 2.2 Multi-Factor Authentication (MFA)
* MFA via hardware token (YubiKey) or approved authenticator application (Google Authenticator / Okta Verify) is **mandatory** for all corporate logins, Single Sign-On (SSO) portals, and VPN endpoints.
* SMS-based 2FA is explicitly deprecated and prohibited due to SIM-swapping vulnerabilities.

---

## 3. Device & Endpoint Security

### 3.1 Removable Storage & USB Media
* The use of external USB flash drives, memory cards, and unencrypted external hard drives on corporate workstations is **strictly prohibited**.
* USB data transfer ports are disabled at the OS level via MDM profile. Exemptions require formal CISO clearance and cryptographically verified storage.

### 3.2 Clean Screen & Inactivity Lock
* All laptops and workstations must automatically lock after **5 minutes** of inactivity.
* Employees must manually lock screens (Windows + L / Control + Command + Q) whenever leaving their work area unattended.

### 3.3 Remote Network Connection (VPN)
* Connecting to Sentinel internal resources from public or untrusted networks (airports, cafes, hotels) requires active connection to the corporate **WireGuard / GlobalProtect VPN**.
* Split-tunneling is strictly disabled.
