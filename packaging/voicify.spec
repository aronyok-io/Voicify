%global debug_package %{nil}

Name:           voicify
Version:        1.0.0
Release:        1%{?dist}
Summary:        Local voice conversion and virtual microphone for Linux
License:        MIT
URL:            https://github.com/aronyok-io/Voicify
Source0:        %{name}-%{version}.tar.gz
ExclusiveArch:  x86_64

BuildRequires:  desktop-file-utils
Requires:       bash
Requires:       coreutils
Requires:       python3-pyside6
Requires:       pipewire-utils
Requires:       pipewire-pulseaudio
Requires:       pulseaudio-utils
Requires:       wireplumber
Requires:       uv

%description
Voicify Studio turns a physical microphone into a virtual microphone for
calls, recording, and games. It can pass through the original voice or
run a local RVC ONNX voice model.

%prep
%autosetup

%build

%install
install -d %{buildroot}%{_libexecdir}/voicify/voicify %{buildroot}%{_libexecdir}/voicify/assets
install -m 0644 voicify/*.py %{buildroot}%{_libexecdir}/voicify/voicify/
install -m 0644 pyproject.toml uv.lock %{buildroot}%{_libexecdir}/voicify/
install -m 0644 assets/voicify.svg %{buildroot}%{_libexecdir}/voicify/assets/
install -m 0755 launch setup %{buildroot}%{_libexecdir}/voicify/
install -m 0755 packaging/voicify %{buildroot}%{_libexecdir}/voicify/run
install -d %{buildroot}%{_bindir}
ln -s ../libexec/voicify/run %{buildroot}%{_bindir}/voicify
desktop-file-install --dir=%{buildroot}%{_datadir}/applications packaging/voicify.desktop
install -Dm0644 assets/voicify.svg %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/voicify.svg

%check
desktop-file-validate packaging/voicify.desktop

%files
%license LICENSE
%doc README.md THIRD_PARTY_NOTICES.md docs/
%{_bindir}/voicify
%{_libexecdir}/voicify/
%{_datadir}/applications/voicify.desktop
%{_datadir}/icons/hicolor/scalable/apps/voicify.svg

%changelog
* Sun Nov 08 2026 Voicify contributors - 1.0.0-1
- First Fedora package.
