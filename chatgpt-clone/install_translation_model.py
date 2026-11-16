"""Install the Argos Translate Vietnamese-to-English model for RAG."""
import argostranslate.package
import argostranslate.translate


def main() -> None:
    installed = argostranslate.package.get_installed_packages()
    if any(package.from_code == "vi" and package.to_code == "en" for package in installed):
        print("Argos Translate vi -> en model is already installed.")
        return
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    package = next((item for item in packages if item.from_code == "vi" and item.to_code == "en"), None)
    if package is None:
        raise RuntimeError("The Argos Translate vi -> en model is unavailable in the package index.")
    argostranslate.package.install_from_path(package.download())
    print("Installed Argos Translate vi -> en model.")


if __name__ == "__main__":
    main()
