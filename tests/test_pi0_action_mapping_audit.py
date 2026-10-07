import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.check_pi0_action_mapping import audit_dataset


@pytest.mark.parametrize('source,semantic', [('action.joint', 'joint'), ('action.ee_pose', 'eepose')])
def test_alias_audit_checks_all_files_and_preserves_data(tmp_path, source, semantic):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'data/chunk-000').mkdir(parents=True)
    (tmp_path / 'meta/info.json').write_text(json.dumps({'fps': 30, 'features': {'action': {'shape': [7]}}}))
    q = np.arange(21).reshape(3, 7).astype(float)
    ee = q / 10.
    paths = []
    for i in range(2):
        path = tmp_path / f'data/chunk-000/episode_{i:06d}.parquet'
        pq.write_table(pa.table({'action.joint': q.tolist(), 'action.ee_pose': ee.tolist(),
                                'action': (q if source == 'action.joint' else ee).tolist()}), path)
        paths.append(path)
    before = [path.read_bytes() for path in paths]
    result = audit_dataset(tmp_path)
    assert result['data_semantic'] == semantic
    assert result['exact_matching_columns'] == [source]
    assert result['rows_checked'] == 6
    assert before == [path.read_bytes() for path in paths]
    # The first file alone would misclassify a dataset with inconsistent aliases.
    pq.write_table(pa.table({'action.joint': q.tolist(), 'action.ee_pose': ee.tolist(),
                            'action': (q + 1).tolist()}), paths[-1])
    assert audit_dataset(tmp_path)['data_semantic'] == 'unverified'


def test_names_and_seven_dimensions_alone_do_not_prove_semantics(tmp_path):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'meta/info.json').write_text('{}')
    pq.write_table(pa.table({'action': [[1.] * 7]}), tmp_path / 'data/episode.parquet')
    assert audit_dataset(tmp_path)['data_semantic'] == 'unverified'
    with pytest.raises(ValueError, match='absent'):
        audit_dataset(tmp_path, 'action.joint')
