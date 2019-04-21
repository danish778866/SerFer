import configparser

class SerferConfig:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def is_section_present(self, section):
        present = False
        if section in self.config:
            present = True
        return present

    def get_section(self, section):
        section_data = None
        if self.is_section_present(section):
            section_data = self.config[section]
        return section_data

#s = SerferConfig("serfer.conf.example")
#print(s.get_section("Lambda")["fn_names"])
