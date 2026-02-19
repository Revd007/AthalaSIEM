# Athala SIEM Agent — Installer (Commercial Distribution)

Installer untuk mendistribusikan Athala SIEM Agent ke lingkungan Windows dan Linux. Cocok untuk penjualan produk dan deployment enterprise.

## Ringkasan

| Platform | Format        | Lokasi              | Kebutuhan build        |
|----------|---------------|---------------------|-------------------------|
| Windows  | MSI (WiX)     | `Installer/Windows/`| WiX Toolset v4/v5, PS   |
| Linux    | DEB, RPM      | `Installer/Linux/`  | dotnet, dpkg-deb/rpmbuild |

---

## Windows MSI Installer

### Persyaratan

- **PowerShell** (Run as Administrator untuk build MSI)
- **WiX Toolset v4 atau v5** — [https://wixtoolset.org/](https://wixtoolset.org/)
- **.NET 8 SDK** (untuk publish agent)

### Build MSI

1. Buka PowerShell **as Administrator** (disarankan).
2. Masuk ke folder installer Windows:
   ```powershell
   cd agent\Installer\Windows
   ```
3. Jalankan script build:
   ```powershell
   .\build-installer.ps1
   ```
   Script akan:
   - Menerbitkan agent (`dotnet publish -c Release -r win-x64`) jika folder publish belum ada
   - Menyalin `AthalaSIEM.Agent.exe` dan `appsettings.json` ke `source\`
   - Membangun MSI dengan WiX
4. Output: **`AthalaSIEMAgent.msi`** di folder `Installer\Windows\`.

### Instalasi oleh end-user

- **Jalankan MSI as Administrator** (klik kanan → Run as administrator).
- Ikuti wizard: Server IP (mis. `http://siem.company.com:9595`), Agent Name, Deployment Token (opsional), Use SSL.
- Backend REST = port 9595 (default), gRPC = port 50051. Installer akan mengisi **BackendGrpcUrl** otomatis (50051) jika Server URL memakai port 9595.
- Setelah selesai: service **Athala SIEM Agent** terpasang (auto start), bisa dijalankan dari Services (`services.msc`) atau Start Menu shortcut.

### Custom action (saat install)

Installer mengubah **appsettings.json** sesuai input wizard:

- **Agent.BackendApiUrl** — URL backend (REST, port 9595)
- **Agent.BackendGrpcUrl** — URL gRPC (port 50051 jika REST 9595)
- **Agent.AgentName** — Nama agent
- **Agent.DeploymentToken** — Token deployment (jika diisi)

### Silent install (opsional)

```powershell
msiexec /i AthalaSIEMAgent.msi /quiet SERVERURL="http://siem.company.com:9595" NAME="Workstation-01" TOKEN="your-deployment-token" SILENT=1
```

---

## Linux (DEB / RPM)

### Build

```bash
cd agent/Installer/Linux
chmod +x build.sh build-deb.sh build-rpm.sh
./build.sh --all
```

Output di subfolder `build/` (paket .deb dan/atau .rpm).

### Persyaratan

- .NET 8 SDK
- `dpkg-deb` (untuk DEB), `rpmbuild` (untuk RPM)

---

## Checklist untuk penjualan / distribusi

- [ ] **Versi & branding**: Sesuaikan `Version`, `Manufacturer`, `UpgradeCode` di `AgentInstaller.wxs` (dan metadata di Linux scripts jika ada).
- [ ] **Sertifikasi kode**: Pertimbangkan code signing untuk MSI dan executable (mis. signtool / sertifikat EV) agar Windows SmartScreen tidak memperingatkan.
- [ ] **Lisensi**: Tambah dialog atau file lisensi di installer jika produk komersial berlisensi.
- [ ] **Dokumentasi**: Sertakan dokumen “Installation Guide” dan “Configuration Guide” untuk pelanggan.
- [ ] **Backend**: Pastikan pelanggan menjalankan backend dengan gRPC di port **50051** dan REST di **9595** (atau sesuai yang dikonfigurasi).

---

## Struktur folder setelah build (Windows)

```
Installer/Windows/
  source/
    AthalaSIEM.Agent.exe   # dari dotnet publish
    appsettings.json       # dari dotnet publish (akan di-patch saat install)
  AgentInstaller.wxs       # definisi WiX
  build-installer.ps1      # script build
  AthalaSIEMAgent.msi      # hasil build (untuk distribusi)
```

---

## Troubleshooting

- **“Build directory not found”**  
  Jalankan dari folder `Installer\Windows`; script akan coba menjalankan `dotnet publish` dari folder project agent. Pastikan .NET 8 SDK terpasang.

- **“WiX Toolset not found”**  
  Instal WiX (v4 atau v5) dan pastikan `candle`/`light` atau `wix` ada di PATH.

- **Service tidak terpasang**  
  Jalankan MSI **as Administrator**.

- **Agent tidak konek ke backend**  
  Cek **BackendGrpcUrl** (port 50051) dan **BackendApiUrl** (port 9595) di `appsettings.json` setelah install (di folder instalasi agent).
