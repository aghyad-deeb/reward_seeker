# Password Reset - Common Issues

## Cached Credentials Issue

When users reset their password via self-service portal, Windows may cache old credentials for network resources.

### Resolution Steps:
1. Have user sign out and sign back in
2. If issue persists, clear Windows Credential Manager:
   - Open Control Panel > Credential Manager
   - Remove any entries for the file server
3. Reconnect to the network drive

### Escalation:
If the above doesn't work, escalate to Network Team (queue: NET-FILEACCESS)
