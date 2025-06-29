from data_loader import load_data

def test_data_shape():
    df = load_data("dataset/iris.csv")
    assert df.shape[1] == 5
    assert 'species' in df.columns

def test_first_species_value():
    df = load_data("dataset/iris.csv")
    assert df.iloc[0]['species'] == 'setosa'
