const google = require('googleapis');

exports.helloPubSub = (event, context) => {
  // in this case the PubSub message payload and attributes are not used
  // but can be used to pass parameters needed by the Dataflow template
  const pubsubMessage = event.data;
  console.log(Buffer.from(pubsubMessage, 'base64').toString());
  console.log(event.attributes);

  google.google.auth.getApplicationDefault(function (err, authClient, projectId) {
  if (err) {
    console.error('Error occurred: ' + err.toString());
    throw new Error(err);
  }

  const dataflow = google.google.dataflow({ version: 'v1b3', auth: authClient });

  dataflow.projects.templates.create({
        projectId: 'dark-pipe-247510',
        resource: {
          parameters: {},
          jobName: 'PipelineDF',
          gcsPath: 'gs://dark-pipe-247510/template1.json'
        }
      }, function(err, response) {
        if (err) {
          console.error("Problem running dataflow template, error was: ", err);
        }
        console.log("Dataflow template response: ", response);
      });
  });
};
