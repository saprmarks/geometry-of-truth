import pandas as pd
import os

ROOT = '~/statement_reps/datasets'

df = pd.read_csv(os.path.join(ROOT, 'geonames.csv'))

countries = {
    'Kenya' : 'Kenya',
    'Croatia' : 'Croatia',
    'Austria' : 'Austria',
    'Peru' : 'Peru',
    'Zambia' : 'Zambia',
    'Tajikistan' : 'Tajikistan',
    'Niger' : 'Niger',
    'Congo, Democratic Republic of the' : 'the Democratic Republic of the Congo',
    'Algeria' : 'Algeria',
    'Trinidad and Tobago' : 'Trinidad and Tobago',
    'Cyprus' : 'Cyprus',
    'Mauritania' : 'Mauritania',
    'Uruguay' : 'Uruguay',
    'Slovenia' : 'Slovenia',
    'Saint Vincent and the Grenadines' : 'Saint Vincent and the Grenadines',
    'Bolivia' : 'Bolivia',
    'Malawi' : 'Malawi',
    'Bangladesh' : 'Bangladesh',
    'Turkey' : 'Turkey',
    'Vanuatu' : 'Vanuatu',
    'Madagascar' : 'Madagascar',
    'Hungary' : 'Hungary',
    'Kyrgyzstan' : 'Kyrgyzstan',
    'New Zealand' : 'New Zealand',
    'Uzbekistan' : 'Uzbekistan',
    'Iran, Islamic Rep. of' : 'Iran',
    'Togo' : 'Togo',
    'Tonga' : 'Tonga',
    'Monaco' : 'Monaco',
    'Luxembourg' : 'Luxembourg',
    'Costa Rica' : 'Costa Rica',
    'Belize' : 'Belize',
    'Montenegro' : 'Montenegro',
    'Spain' : 'Spain',
    'Lebanon' : 'Lebanon',
    'Poland' : 'Poland',
    'Tanzania, United Republic of' : 'Tanzania',
    'Australia' : 'Australia',
    'Angola' : 'Angola',
    'Saint Kitts and Nevis' : 'Saint Kitts and Nevis',
    'Panama' : 'Panama',
    'Samoa' : 'Samoa',
    'Switzerland' : 'Switzerland',
    'Burundi' : 'Burundi',
    'Mozambique' : 'Mozambique',
    'Papua New Guinea' : 'Papua New Guinea',
    'Bulgaria' : 'Bulgaria',
    'Chile' : 'Chile',
    'Mali' : 'Mali',
    'Venezuela, Bolivarian Rep. of' : 'Venezuela',
    'South Africa' : 'South Africa',
    'United Kingdom' : 'the United Kingdom',
    'Comoros' : 'Comoros',
    'Japan' : 'Japan',
    'Albania' : 'Albania',
    'Senegal' : 'Senegal',
    'Guatemala' : 'Guatemala',
    'Guinea' : 'Guinea',
    'Malaysia' : 'Malaysia',
    'Yemen' : 'Yemen',
    'Nauru' : 'Nauru',
    'Syrian Arab Republic' : 'Syria',
    'Slovakia' : 'Slovakia',
    'Germany' : 'Germany',
    'Ecuador' : 'Ecuador',
    'Lithuania' : 'Lithuania',
    'Dominica' : 'Dominica',
    'Azerbaijan' : 'Azerbaijan',
    'Sudan, The Republic of' : 'Sudan',
    'Seychelles' : 'Seychelles',
    'Kiribati' : 'Kiribati',
    'Iraq' : 'Iraq',
    'Namibia' : 'Namibia',
    'Congo' : 'the Republic of the Congo',
    'Andorra' : 'Andorra',
    'Canada' : 'Canada',
    'Korea, Republic of' : 'South Korea',
    'Bahamas' : 'the Bahamas',
    'Sierra Leone' : 'Sierra Leone',
    'Brazil' : 'Brazil',
    'Finland' : 'Finland',
    'Ukraine' : 'Ukraine',
    'Norway' : 'Norway',
    'Russian Federation' : 'Russia',
    'Cuba' : 'Cuba',
    'Sao Tome and Principe' : 'Sao Tome and Principe',
    'Estonia' : 'Estonia',
    'Portugal' : 'Portugal',
    'Greece' : 'Greece',
    'Bhutan' : 'Bhutan',
    'Latvia' : 'Latvia',
    'Central African Republic' : 'the Central African Republic',
    'Zimbabwe' : 'Zimbabwe',
    'Lesotho' : 'Lesotho',
    'Moldova, Republic of' : 'Moldova',
    'Mauritius' : 'Mauritius',
    'Palau' : 'Palau',
    'Nicaragua' : 'Nicaragua',
    'Djibouti' : 'Djibouti',
    'Gabon' : 'Gabon',
    'Dominican Republic' : 'the Dominican Republic',
    'Qatar' : 'Qatar',
    'Bosnia and Herzegovina' : 'Bosnia and Herzegovina',
    'Kazakhstan' : 'Kazakhstan',
    'Maldives' : 'the Maldives',
    'China' : 'China',
    'Viet Nam' : 'Vietnam',
    "Korea, Dem. People's Rep. of" : 'North Korea',
    'Myanmar' : 'Myanmar',
    'Turkmenistan' : 'Turkmenistan',
    'Barbados' : 'Barbados',
    'San Marino' : 'San Marino',
    'Romania' : 'Romania',
    'Armenia' : 'Armenia',
    'United Arab Emirates' : 'the United Arab Emirates',
    'Malta' : 'Malta',
    'Uganda' : 'Uganda',
    'United States' : 'the United States',
    'Saudi Arabia' : 'Saudi Arabia',
    'Ethiopia' : 'Ethiopia',
    'Guyana' : 'Guyana',
    'Benin' : 'Benin',
    'India' : 'India',
    'Macedonia, The former Yugoslav Rep. of' : 'Macedonia',
    'Philippines' : 'the Philippines',
    'Mexico' : 'Mexico',
    'Fiji' : 'Fiji',
    'Bahrain' : 'Bahrain',
    'Belarus' : 'Belarus',
    'Afghanistan' : 'Afghanistan',
    "Côte d'Ivoire" : "Côte d'Ivoire",
    'France' : 'France',
    'Kuwait' : 'Kuwait',
    'Czech Republic' : 'the Czech Republic',
    'Egypt' : 'Egypt',
    'Jordan' : 'Jordan',
    'Gambia' : 'the Gambia',
    'Equatorial Guinea' : 'Equatorial Guinea',
    'Oman' : 'Oman',
    'Denmark' : 'Denmark',
    'Haiti' : 'Haiti',
    'El Salvador' : 'El Salvador',
    'Liberia' : 'Liberia',
    'Tuvalu' : 'Tuvalu',
    'Burkina Faso' : 'Burkina Faso',
    'Chad' : 'Chad',
    'Guinea-Bissau' : 'Guinea-Bissau',
    'Cape Verde' : 'Cape Verde',
    'Somalia' : 'Somalia',
    'Indonesia' : 'Indonesia',
    'Tunisia' : 'Tunisia',
    'Belgium' : 'Belgium',
    'Liechtenstein' : 'Liechtenstein',
    'Colombia' : 'Colombia',
    "Lao People's Dem. Rep." : 'Laos',
    'Timor-Leste' : 'Timor-Leste',
    'Honduras' : 'Honduras',
    'Italy' : 'Italy',
    'Serbia' : 'Serbia',
    'Netherlands' : 'the Netherlands',
    'Jamaica' : 'Jamaica',
    'Eritrea' : 'Eritrea',
    'Nepal' : 'Nepal',
    'Swaziland' : 'Swaziland',
    'Antigua and Barbuda' : 'Antigua and Barbuda',
    'Rwanda' : 'Rwanda',
    'Paraguay' : 'Paraguay',
    'Sri Lanka' : 'Sri Lanka',
    'Iceland' : 'Iceland',
    'Morocco' : 'Morocco',
    'Suriname' : 'Suriname',
    'Argentina' : 'Argentina',
    'Mongolia' : 'Mongolia',
    'Botswana' : 'Botswana',
    'Thailand' : 'Thailand',
    'Cameroon' : 'Cameroon',
    'Ireland' : 'Ireland',
    'Nigeria' : 'Nigeria',
    'Cambodia' : 'Cambodia',
    'Sweden' : 'Sweden',
    'Pakistan' : 'Pakistan',
    'Ghana' : 'Ghana',
    'Singapore' : 'Singapore',
}

def is_valid(row):
    name = row['ASCII Name']
    population = row['Population']
    country = row['Country name EN']
    
    # check that the country is valid
    if country not in countries:
        return False
    
    # check population is larger than 500000
    if not population > 500000:
        return False
    
    # check that not a city-state
    if name in country:
        return False
    
    # check there is no other city with the same name
    if len(df[df['ASCII Name'] == name]) > 1:
        return False
    
    return True

df = df[df.apply(is_valid, axis=1)]
df_out = {
    'statement' : [],
    'label' : [],
    'city' : [],
    'country' : [],
    'correct_country' : [],
}

neg_df_out = {
    'statement' : [],
    'label' : [],
    'city' : [],
    'country' : [],
    'correct_country' : [],
}

for _, row in df.iterrows():
    city = row['ASCII Name']
    country = countries[row['Country name EN']]

    # make true statement
    statement = f'The city of {city} is in {country}.'
    neg_statement = f'The city of {city} is not in {country}.'
    df_out['statement'].append(statement), neg_df_out['statement'].append(neg_statement)
    df_out['label'].append(1), neg_df_out['label'].append(0)
    df_out['city'].append(city), neg_df_out['city'].append(city)
    df_out['country'].append(country), neg_df_out['country'].append(country)
    df_out['correct_country'].append(country), neg_df_out['correct_country'].append(country)

    # make false statement
    false_country = countries[df['Country name EN'].sample(1).iloc[0]]
    while false_country == country:
        false_country = countries[df['Country name EN'].sample(1).iloc[0]]
    statement = f'The city of {city} is in {false_country}.'
    neg_statement = f'The city of {city} is not in {false_country}.'
    df_out['statement'].append(statement), neg_df_out['statement'].append(neg_statement)
    df_out['label'].append(0), neg_df_out['label'].append(1)
    df_out['city'].append(city), neg_df_out['city'].append(city)
    df_out['country'].append(false_country), neg_df_out['country'].append(false_country)
    df_out['correct_country'].append(country), neg_df_out['correct_country'].append(country)

df_out, neg_def_out = pd.DataFrame(df_out), pd.DataFrame(neg_df_out)
df_out.to_csv(os.path.join(ROOT, 'cities.csv'), index=False), neg_def_out.to_csv(os.path.join(ROOT, 'neg_cities.csv'), index=False)
