# TraceCMR shortcuts on the Desktop and in the Start menu, with the TraceCMR logo.
#
# Each shortcut carries TraceCMR's app ID (System.AppUserModel.ID = TraceCMR.TraceCMR, the same ID the app
# registers when it starts). Windows uses it to tie the running app to the shortcut, so the taskbar button shows
# the logo (not a blank icon) and "Pin to taskbar" pins TraceCMR itself. Called by "TraceCMR update.bat".
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File make_shortcut.ps1 -Target ...\TraceCMR.bat -Icon ...\tracecmr.ico

param(
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(Mandatory = $true)][string]$Icon,
    [string]$WorkDir = (Split-Path -Parent $Target),
    [string]$AppId = "TraceCMR.TraceCMR",
    [string]$Description = "TraceCMR: mouse cardiac MRI analysis"
)

$code = @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

public static class TraceCmrShortcut
{
    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    private class CShellLink { }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
    private interface IShellLinkW
    {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszFile, int cch, IntPtr pfd, uint fFlags);
        void GetIDList(out IntPtr ppidl);
        void SetIDList(IntPtr pidl);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszName, int cch);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string pszName);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszDir, int cch);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string pszDir);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszArgs, int cch);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string pszArgs);
        void GetHotkey(out short pwHotkey);
        void SetHotkey(short wHotkey);
        void GetShowCmd(out int piShowCmd);
        void SetShowCmd(int iShowCmd);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszIconPath, int cch, out int piIcon);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string pszIconPath, int iIcon);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string pszPathRel, uint dwReserved);
        void Resolve(IntPtr hwnd, uint fFlags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string pszFile);
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    private struct PropertyKey
    {
        public Guid fmtid;
        public uint pid;
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct PropVariant
    {
        [FieldOffset(0)] public ushort vt;
        [FieldOffset(8)] public IntPtr pointer;
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    private interface IPropertyStore
    {
        void GetCount(out uint cProps);
        void GetAt(uint iProp, out PropertyKey pkey);
        void GetValue(ref PropertyKey key, out PropVariant pv);
        void SetValue(ref PropertyKey key, ref PropVariant pv);
        void Commit();
    }

    public static void Save(string lnkPath, string target, string workDir, string icon, string appId,
                            string description, int showCmd)
    {
        IShellLinkW link = (IShellLinkW)new CShellLink();
        link.SetPath(target);
        link.SetWorkingDirectory(workDir);
        link.SetIconLocation(icon, 0);
        link.SetDescription(description);
        link.SetShowCmd(showCmd);

        IPropertyStore store = (IPropertyStore)link;
        PropertyKey key = new PropertyKey();
        key.fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");  // System.AppUserModel.ID
        key.pid = 5;
        PropVariant value = new PropVariant();
        value.vt = 31;  // VT_LPWSTR
        value.pointer = Marshal.StringToCoTaskMemUni(appId);
        try
        {
            store.SetValue(ref key, ref value);
            store.Commit();
        }
        finally
        {
            Marshal.FreeCoTaskMem(value.pointer);
        }
        ((IPersistFile)link).Save(lnkPath, true);
    }
}
"@

$places = @(
    [Environment]::GetFolderPath('Desktop'),
    [Environment]::GetFolderPath('Programs')  # Start menu: where Windows looks up app IDs for the taskbar
)
$ok = $true
try {
    Add-Type -TypeDefinition $code -Language CSharp -ErrorAction Stop
    foreach ($dir in $places) {
        $lnk = Join-Path $dir 'TraceCMR.lnk'
        [TraceCmrShortcut]::Save($lnk, $Target, $WorkDir, $Icon, $AppId, $Description, 7)  # 7: start minimised
        Write-Output "shortcut with app ID: $lnk"
    }
}
catch {
    # fall back to a plain shortcut (logo, no app ID) so there is always something to double-click
    Write-Output "app ID shortcut failed: $($_.Exception.Message)"
    $ok = $false
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $places[0] 'TraceCMR.lnk'))
    $s.TargetPath = $Target
    $s.WorkingDirectory = $WorkDir
    $s.IconLocation = $Icon
    $s.Description = $Description
    $s.WindowStyle = 7
    $s.Save()
    Write-Output "plain shortcut: $($places[0])\TraceCMR.lnk"
}
if (-not $ok) { exit 2 }
