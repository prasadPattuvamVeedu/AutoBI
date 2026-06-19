from django.db import migrations, models
import django.db.models.deletion


def backfill_version_types(apps, schema_editor):
    DatasetVersion = apps.get_model("datasets", "DatasetVersion")
    for version in DatasetVersion.objects.all():
        version.version_type = "cleaned" if version.is_cleaned else "original"
        version.is_active = True
        version.save(update_fields=["version_type", "is_active"])


class Migration(migrations.Migration):

    dependencies = [
        ("datasets", "0004_datasetversion_transformation_log"),
    ]

    operations = [
        migrations.AddField(
            model_name="datasetversion",
            name="version_type",
            field=models.CharField(
                choices=[
                    ("original", "Original"),
                    ("cleaned", "Cleaned"),
                    ("feature_engineered", "Feature engineered"),
                    ("ml_ready", "ML ready"),
                ],
                default="original",
                max_length=40,
            ),
        ),
        migrations.AddField(
            model_name="datasetversion",
            name="parent_version",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="child_versions",
                to="datasets.datasetversion",
            ),
        ),
        migrations.AddField(
            model_name="datasetversion",
            name="preview_rows",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="datasetversion",
            name="columns",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="datasetversion",
            name="transformation_plan_json",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="datasetversion",
            name="is_active",
            field=models.BooleanField(default=True),
        ),
        migrations.RunPython(backfill_version_types, migrations.RunPython.noop),
    ]
