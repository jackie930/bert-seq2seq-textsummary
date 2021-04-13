import pandas as pd
import re


def process_summary(summary):
    # test = summary.replace('\n\n','\n').split('\n')
    pattern = re.compile(r'.*?beginbegin([\s\S]*?)endend.*?')
    summary_text = ''.join(re.findall(pattern, summary))
    return summary_text


def main(input_files):
    # convert excel files to txt labels
    summary_length = []
    text_length = []
    summary_percent = []
    df = pd.read_excel(input_files, engine='openpyxl')
    for i in range(len(df)):
        main_file = df['正文'][i]
        summary = df['摘要'][i]
        summary_text = process_summary(summary)
        if len(summary_text) > 0:
            txt_name = '../data/' + str(i) + '.txt'

            with open(txt_name, 'a', encoding='utf-8') as f:
                print ("<<<< process row number: ", i)
                print ("<<<< source text length %d, extracted summary length: %d, percentage %f " % (
                len(summary), len(summary_text), len(summary_text) / len(summary)))
                f.write(summary_text.replace('|', ''))
                f.write('SPLIT')
                f.write(main_file.replace('|', ''))

            f.close()
            summary_length.append(len(summary))
            text_length.append(len(summary_text))
            summary_percent.append(len(summary_text) / len(summary))

    # output df
    df_output = pd.DataFrame(
        {'summary_length': summary_length, 'text_length': text_length, 'summary_percent': summary_percent})
    df_output.to_csv('result.csv', encoding='utf-8', index=False)
    print("process finished!")


if __name__ == '__main__':
    main('../摘要标注_104条.xlsx')

